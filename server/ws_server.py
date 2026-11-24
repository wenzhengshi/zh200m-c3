#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Zh200M-C3 本地 WebSocket 推理服务。

设计要点
- 复用仓库 inference/interactive_c3_747m_kvcache.py 的 KV Cache 实现与采样函数，
  不重新实现那份非标准 RoPE。
- 单模型实例、单并发：4 核 CPU 上 ~12 token/s，并发只会互相拖慢，故用信号量排队。
- 生成在独立线程里跑（CPU 密集），通过 asyncio.Queue 把增量 token 推回事件循环。
- 限流：每 IP 最小间隔 + 全局排队上限 + 单次 token 上限。
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import hmac
import importlib.util
import json
import logging
import os
import secrets
import sqlite3
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import uvicorn
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse

ROOT = Path(__file__).resolve().parent.parent
INFER_DIR = ROOT / "inference"
KV_SCRIPT = INFER_DIR / "interactive_c3_747m_kvcache.py"
MODEL_PT = ROOT / "zh200m-c3-196m-fp32.pt"
TOKENIZER_PATH = ROOT / "tokenizer_frozen" / "tokenizer.model"
INDEX_HTML = Path(__file__).resolve().parent / "index.html"

ADMIN_HTML = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>评论管理 · Zh200M-C3</title>
<style>
  :root { color-scheme: light; }
  * { box-sizing: border-box; }
  body {
    margin: 0; padding: 24px 16px 60px; background: #f7f8fa; color: #1c2026;
    font: 14px/1.6 -apple-system, "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
  }
  .wrap { max-width: 760px; margin: 0 auto; }
  h1 { font-size: 18px; margin: 0 0 4px; }
  .sub { color: #8b93a0; font-size: 12.5px; margin-bottom: 20px; }
  .card { background: #fff; border: 1px solid #e8eaed; border-radius: 12px; padding: 16px; margin-bottom: 14px; }
  input[type=password], input[type=text] {
    width: 100%; padding: 10px 12px; border: 1px solid #dfe3e8; border-radius: 8px;
    font: inherit; background: #fff; color: #1c2026;
  }
  button {
    font: inherit; padding: 8px 16px; border-radius: 8px; border: 0; cursor: pointer;
    background: #4d6bfe; color: #fff;
  }
  button.ghost { background: #f2f4f7; color: #4a5057; }
  button.danger { background: #fdecec; color: #d33a3a; }
  button:disabled { opacity: .55; cursor: not-allowed; }
  .row { display: flex; gap: 10px; align-items: center; }
  .row > input { flex: 1; }
  .cm { border-top: 1px solid #f0f2f4; padding: 14px 0; display: flex; gap: 12px; }
  .cm:first-child { border-top: 0; }
  .cm .body { flex: 1; min-width: 0; }
  .cm .head { font-size: 12px; color: #9aa0a6; margin-bottom: 4px; }
  .cm .head b { color: #2b2f36; font-weight: 600; font-size: 13px; }
  .cm .text { white-space: pre-wrap; word-break: break-word; font-size: 14px; }
  .empty { color: #9aa0a6; text-align: center; padding: 30px 0; font-size: 13px; }
  .hint { color: #8b93a0; font-size: 12px; margin-top: 10px; }
  .err { color: #d33a3a; font-size: 13px; margin-top: 8px; min-height: 18px; }
  .ut { width: 100%; border-collapse: collapse; margin-top: 10px; font-size: 12.5px; }
  .ut th, .ut td { text-align: left; padding: 6px 8px; border-bottom: 1px solid #f0f2f4; }
  .ut th { color: #8b93a0; font-weight: 500; font-size: 11.5px; }
  .ut td.n { text-align: right; font-variant-numeric: tabular-nums; }
  .ut button { padding: 3px 10px; font-size: 11.5px; }
  .ug { font-size: 12.5px; color: #4a5057; margin-top: 10px; }
  .ug b { color: #1c2026; }
</style>
</head>
<body>
<div class="wrap">
  <h1>评论管理</h1>
  <div class="sub">Zh200M-C3 交流讨论区 · 仅管理员可访问</div>

  <div class="card" id="loginBox" style="display:none">
    <div class="row">
      <input type="password" id="pwd" placeholder="管理员密码" autocomplete="current-password">
      <button id="loginBtn">登录</button>
    </div>
    <div class="err" id="loginErr"></div>
  </div>

  <div class="card" id="listBox" style="display:none">
    <div class="row" style="justify-content:space-between">
      <div style="font-size:13px;color:#4a5057">共 <b id="cnt">0</b> 条</div>
      <div class="row">
        <button class="ghost" id="refresh">刷新</button>
        <button class="ghost" id="logout">退出</button>
      </div>
    </div>
    <div id="list"></div>
  </div>

  <div class="card" id="usageBox" style="display:none">
    <div class="row" style="justify-content:space-between">
      <div style="font-size:13px;color:#4a5057">今日用量 <span id="uDay"></span></div>
      <button class="ghost" id="resetAll">全部清零</button>
    </div>
    <div class="ug" id="uGlobal"></div>
    <table class="ut" id="uTable"></table>
  </div>
</div>
<script>
const $ = id => document.getElementById(id);
const esc = s => String(s);

async function api(path, opts) {
  const r = await fetch(path, Object.assign({ credentials: 'same-origin' }, opts || {}));
  let data = {};
  try { data = await r.json(); } catch (e) {}
  if (!r.ok) throw new Error(data.error || ('HTTP ' + r.status));
  return data;
}

function fmt(ts) {
  const d = new Date(ts * 1000), p = x => String(x).padStart(2, '0');
  return d.getFullYear() + '-' + p(d.getMonth() + 1) + '-' + p(d.getDate()) + ' '
    + p(d.getHours()) + ':' + p(d.getMinutes());
}

async function load() {
  const data = await api('/admin/api/comments');
  $('cnt').textContent = data.comments.length;
  const box = $('list');
  box.innerHTML = '';
  if (!data.comments.length) {
    box.innerHTML = '<div class="empty">还没有留言</div>';
    return;
  }
  data.comments.forEach(c => {
    const el = document.createElement('div');
    el.className = 'cm';
    const body = document.createElement('div');
    body.className = 'body';
    const head = document.createElement('div');
    head.className = 'head';
    const name = document.createElement('b');
    name.textContent = c.nickname;
    head.appendChild(name);
    head.appendChild(document.createTextNode(' · ' + fmt(c.created_at) + ' · ' + c.ip));
    const text = document.createElement('div');
    text.className = 'text';
    text.textContent = c.content;
    body.appendChild(head);
    body.appendChild(text);
    const btn = document.createElement('button');
    btn.className = 'danger';
    btn.textContent = '删除';
    btn.onclick = async () => {
      if (!confirm('确定删除这条留言？')) return;
      btn.disabled = true;
      try { await api('/admin/api/comments/' + c.id, { method: 'DELETE' }); await load(); }
      catch (e) { alert(e.message); btn.disabled = false; }
    };
    el.appendChild(body);
    el.appendChild(btn);
    box.appendChild(el);
  });
}

async function loadUsage() {
  const data = await api('/admin/api/usage');
  $('uDay').textContent = '(' + data.day + ')';
  $('uGlobal').innerHTML = '全站请求 <b>' + data.global.requests + '</b> / ' + data.global.request_limit
    + ' · 全站 token <b>' + data.global.tokens + '</b> / ' + data.global.token_limit
    + ' · 单 IP 上限 ' + data.limits.req_per_ip + ' 次 / ' + data.limits.tokens_per_ip + ' token';
  const t = $('uTable');
  t.innerHTML = '';
  if (!data.rows.length) {
    t.innerHTML = '<tr><td colspan="4" style="color:#9aa0a6">今天还没有使用记录</td></tr>';
    return;
  }
  const head = document.createElement('tr');
  ['IP', '请求数', 'token', ''].forEach((h, i) => {
    const th = document.createElement('th');
    th.textContent = h;
    if (i === 1 || i === 2) th.style.textAlign = 'right';
    head.appendChild(th);
  });
  t.appendChild(head);

  data.rows.forEach(r => {
    const tr = document.createElement('tr');
    const ip = document.createElement('td');
    ip.textContent = r.ip;
    const nreq = document.createElement('td');
    nreq.className = 'n';
    nreq.textContent = r.requests;
    const ntok = document.createElement('td');
    ntok.className = 'n';
    ntok.textContent = r.tokens;
    const act = document.createElement('td');
    const btn = document.createElement('button');
    btn.className = 'danger';
    btn.textContent = '清零';
    btn.onclick = async () => {
      if (!confirm('清零 ' + r.ip + ' 的今日用量？')) return;
      await api('/admin/api/usage/' + encodeURIComponent(r.ip), { method: 'DELETE' });
      await loadUsage();
    };
    act.appendChild(btn);
    tr.append(ip, nreq, ntok, act);
    t.appendChild(tr);
  });
}

async function start() {
  try {
    const s = await api('/admin/api/session');
    if (s.admin) {
      $('loginBox').style.display = 'none';
      $('listBox').style.display = 'block';
      $('usageBox').style.display = 'block';
      await load();
      await loadUsage();
    } else {
      $('loginBox').style.display = 'block';
      $('listBox').style.display = 'none';
      $('usageBox').style.display = 'none';
    }
  } catch (e) {
    $('loginBox').style.display = 'block';
  }
}

$('resetAll').onclick = async () => {
  if (!confirm('清零今天全站所有 IP 的用量？')) return;
  await api('/admin/api/usage/reset', { method: 'POST' });
  await loadUsage();
};

$('loginBtn').onclick = async () => {
  $('loginErr').textContent = '';
  $('loginBtn').disabled = true;
  try {
    await api('/admin/login', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ password: $('pwd').value }),
    });
    $('pwd').value = '';
    await start();
  } catch (e) {
    $('loginErr').textContent = e.message;
  } finally {
    $('loginBtn').disabled = false;
  }
};
$('pwd').addEventListener('keydown', e => { if (e.key === 'Enter') $('loginBtn').click(); });
$('refresh').onclick = load;
$('logout').onclick = async () => { await api('/admin/logout', { method: 'POST' }); await start(); };

start();
</script>
</body>
</html>
"""

MAX_PROMPT_CHARS = 500
DEFAULT_MAX_TOKENS = 64
HARD_MAX_TOKENS = 128


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


# ---- 使用限额（可用环境变量覆盖，便于压测） ----
DAILY_REQ_PER_IP = _env_int("ZH_LIMIT_REQ_PER_IP", 120)      # 单 IP 每天请求数
DAILY_TOKENS_PER_IP = _env_int("ZH_LIMIT_TOKENS_PER_IP", 12000)  # 单 IP 每天生成 token
DAILY_REQ_GLOBAL = _env_int("ZH_LIMIT_REQ_GLOBAL", 3000)      # 全站每天请求数
DAILY_TOKENS_GLOBAL = _env_int("ZH_LIMIT_TOKENS_GLOBAL", 400000)  # 全站每天 token
MAX_WS_PER_IP = _env_int("ZH_LIMIT_WS_PER_IP", 3)             # 单 IP 同时连接数
MIN_INTERVAL = 2.0                                            # 两次生成的最小间隔(秒)

_conn_count: dict[str, int] = {}

log = logging.getLogger("zh200m")

# ---------------------------------------------------------------- 运行时状态


class Runtime:
    """全局推理资源。只有一个生成回合能在跑。"""

    def __init__(self) -> None:
        self.mod: Any = None
        self.model: Any = None
        self.tokenizer: Any = None
        self.config: dict[str, Any] = {}
        self.controller: Any = None
        self.generator: Any = None
        self.sem = asyncio.Semaphore(1)
        self.active = 0
        self.waiting = 0
        self.served = 0
        self.generated_tokens = 0
        self.started_at = time.time()
        self.last_by_ip: dict[str, float] = defaultdict(float)

    # ------------------------------------------------------------ 加载
    def load(self) -> None:
        t0 = time.perf_counter()

        spec = importlib.util.spec_from_file_location("zh_kv_runtime", str(KV_SCRIPT))
        if spec is None or spec.loader is None:
            raise RuntimeError(f"无法加载 KV 运行时: {KV_SCRIPT}")
        mod = importlib.util.module_from_spec(spec)
        sys.modules["zh_kv_runtime"] = mod
        spec.loader.exec_module(mod)
        self.mod = mod

        pkg = torch.load(MODEL_PT, map_location="cpu", weights_only=False)
        self.config = dict(pkg["config"])
        state_dict = pkg["state_dict"]
        del pkg

        native = mod.load_python_module(mod.MODEL_PY)
        model = native.build_model(self.config).float().cpu().eval()
        model.load_state_dict(state_dict, strict=True)
        del state_dict
        self.model = model

        import sentencepiece as spm

        tokenizer = spm.SentencePieceProcessor()
        if not tokenizer.load(str(TOKENIZER_PATH)):
            raise RuntimeError("tokenizer 加载失败")
        self.tokenizer = tokenizer

        self.controller = mod.KVCacheController(model, self.config)
        # 内部会 install() 并在通过后 reset()；不通过会抛异常拒绝启动。
        mod.validate_kv_cache(model, tokenizer, self.controller)

        self.generator = mod.CachedGenerator(
            model=model, tokenizer=tokenizer, config=self.config, cache=self.controller
        )
        params = sum(p.numel() for p in model.parameters())
        log.info("模型就绪：%s 参数，%.2fs", f"{params:,}", time.perf_counter() - t0)

    # ------------------------------------------------------------ 生成
    def generate(self, prompt: str, max_tokens: int, temperature: float,
                 top_k: int, top_p: float, rep: float, emit) -> dict[str, Any]:
        """阻塞式生成；每产出一段增量文本调用 emit(delta)。"""
        mod = self.mod
        gen = self.generator
        tok = self.tokenizer

        prompt_ids = tok.encode(prompt, out_type=int)
        if not prompt_ids:
            raise ValueError("提示词编码后为空")
        room = self.config["max_seq_len"] - max_tokens - 1
        if room <= 0:
            raise ValueError("token 上限过大")
        if len(prompt_ids) > room:
            prompt_ids = prompt_ids[-room:]
            emit("")

        t_start = time.perf_counter()
        gen.rebuild(prompt_ids)
        ttft = None

        generated: list[int] = []
        streamed = ""
        eos = int(tok.eos_id())

        for _ in range(max_tokens):
            next_id = mod.sample_next_token(
                gen.next_logits,
                gen.token_ids,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=rep,
            )
            if eos >= 0 and next_id == eos:
                break
            generated.append(next_id)
            current = tok.decode(generated)
            if current.startswith(streamed):
                delta = current[len(streamed):]
            else:
                delta = tok.decode([next_id])
            streamed = current
            if delta:
                if ttft is None:
                    ttft = time.perf_counter() - t_start
                emit(delta)
            gen.append_one(next_id)

        elapsed = time.perf_counter() - t_start
        return {
            "text": tok.decode(generated),
            "tokens": len(generated),
            "ttft": round(ttft if ttft is not None else elapsed, 3),
            "elapsed": round(elapsed, 3),
            "tok_per_s": round(len(generated) / elapsed, 2) if elapsed > 0 else 0.0,
        }


RT = Runtime()
app = FastAPI(title="Zh200M-C3 WS API", docs_url=None, redoc_url=None)


# ---------------------------------------------------------------- HTTP 路由


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    return HTMLResponse(INDEX_HTML.read_text(encoding="utf-8"))


@app.get("/healthz", response_class=PlainTextResponse)
async def healthz() -> str:
    return "ok"


@app.get("/stats")
async def stats() -> JSONResponse:
    g_req, g_tok = usage_global()
    return JSONResponse(
        {
            "model": "zh200m-c3-196m",
            "parameters": 196_153_216,
            "uptime_s": round(time.time() - RT.started_at, 1),
            "active": RT.active,
            "waiting": RT.waiting,
            "served": RT.served,
            "generated_tokens": RT.generated_tokens,
            "max_tokens_per_request": HARD_MAX_TOKENS,
            "today": {
                "requests": g_req,
                "request_limit": DAILY_REQ_GLOBAL,
                "tokens": g_tok,
                "token_limit": DAILY_TOKENS_GLOBAL,
            },
        }
    )


@app.get("/api/quota")
async def api_quota(request: Request) -> JSONResponse:
    ip = _client_ip(request)
    q = quota_status(ip)
    q["ip"] = ip
    return JSONResponse(q)


@app.get("/api/info")
async def api_info() -> JSONResponse:
    return JSONResponse(
        {
            "name": "Zh200M-C3",
            "description": "196M 参数中文 Decoder-only 基础模型（base model，非对话模型）",
            "websocket": "/ws",
            "protocol": {
                "request": {
                    "type": "generate",
                    "prompt": "中国的首都是",
                    "max_tokens": DEFAULT_MAX_TOKENS,
                    "temperature": 0.9,
                    "top_k": 40,
                    "top_p": 0.9,
                    "repetition_penalty": 1.2,
                },
                "events": ["queued", "start", "token", "done", "error"],
            },
            "stateless": True,
            "hardware": {
                "device": "cpu",
                "cpu": "Intel Core i5-4460 @ 3.20GHz",
                "cores": 4,
                "memory_gb": 16,
                "gpu": False,
                "dtype": "float32",
                "quantized": False,
                "model_file_mb": 784,
                "process_rss_mb": 1035,
                "measured_tokens_per_sec": "12-15",
                "runtime": "Python 3.13 / PyTorch 2.13.0+cpu",
            },
            "note": "基础模型只会续写，不保证事实正确性；每个请求互相独立，"
                    "服务端不保存任何对话上下文，多轮记忆需由客户端自行拼接。",
        }
    )


# ---------------------------------------------------------------- WebSocket


def _client_ip(ws: WebSocket) -> str:
    fwd = ws.headers.get("x-real-ip") or ws.headers.get("x-forwarded-for")
    if fwd:
        return fwd.split(",")[0].strip()
    return ws.client.host if ws.client else "unknown"


async def _pump(ws: WebSocket, prompt: str, params: dict[str, Any]) -> dict[str, Any] | None:
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()
    holder: dict[str, Any] = {}
    meta: dict[str, Any] | None = None

    def emit(delta: str) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, ("token", delta))

    def work() -> None:
        try:
            meta = RT.generate(
                prompt,
                params["max_tokens"],
                params["temperature"],
                params["top_k"],
                params["top_p"],
                params["repetition_penalty"],
                emit,
            )
            loop.call_soon_threadsafe(queue.put_nowait, ("done", meta))
        except BaseException as exc:  # noqa: BLE001
            holder["error"] = f"{type(exc).__name__}: {exc}"
            loop.call_soon_threadsafe(queue.put_nowait, ("error", holder["error"]))

    thread = threading.Thread(target=work, daemon=True)
    RT.active += 1
    try:
        await ws.send_json({"type": "start"})
        thread.start()
        while True:
            kind, payload = await queue.get()
            if kind == "token":
                await ws.send_json({"type": "token", "text": payload})
            elif kind == "done":
                RT.served += 1
                RT.generated_tokens += payload["tokens"]
                meta = payload
                await ws.send_json({"type": "done", **payload})
                break
            else:
                await ws.send_json({"type": "error", "message": payload})
                break
    finally:
        RT.active -= 1
        # 必须等生成线程真正结束再释放信号量：生成器是全局共享的，
        # 客户端中途断开时若提前放行，另一个请求会踩到同一份 KV Cache。
        if thread.is_alive():
            thread.join()
    return meta


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket) -> None:
    ip = _client_ip(ws)
    if _conn_count.get(ip, 0) >= MAX_WS_PER_IP:
        await ws.accept()
        await ws.send_json({
            "type": "error",
            "message": f"同一 IP 同时最多 {MAX_WS_PER_IP} 个连接",
        })
        await ws.close()
        return

    await ws.accept()
    _conn_count[ip] = _conn_count.get(ip, 0) + 1

    try:
        while True:
            raw = await ws.receive_text()
            try:
                req = json.loads(raw)
            except json.JSONDecodeError:
                await ws.send_json({"type": "error", "message": "请求必须是 JSON"})
                continue

            if req.get("type") == "ping":
                await ws.send_json({"type": "pong", "server_time": time.time()})
                continue

            prompt = str(req.get("prompt", "")).strip()
            if not prompt:
                await ws.send_json({"type": "error", "message": "prompt 不能为空"})
                continue
            if len(prompt) > MAX_PROMPT_CHARS:
                await ws.send_json(
                    {"type": "error", "message": f"prompt 最多 {MAX_PROMPT_CHARS} 字"}
                )
                continue

            now = time.time()
            gap = now - RT.last_by_ip[ip]
            if gap < MIN_INTERVAL:
                await ws.send_json(
                    {"type": "error", "message": f"太快了，请等 {MIN_INTERVAL - gap:.1f}s 再试"}
                )
                continue
            RT.last_by_ip[ip] = now

            params = {
                "max_tokens": max(1, min(int(req.get("max_tokens", DEFAULT_MAX_TOKENS)), HARD_MAX_TOKENS)),
                "temperature": max(0.0, min(float(req.get("temperature", 0.9)), 2.0)),
                "top_k": max(0, min(int(req.get("top_k", 40)), 200)),
                "top_p": max(0.05, min(float(req.get("top_p", 0.9)), 1.0)),
                "repetition_penalty": max(1.0, min(float(req.get("repetition_penalty", 1.2)), 2.0)),
            }

            # ---- 配额检查：先扣次数，生成完再扣 token ----
            quota = quota_status(ip, params["max_tokens"])
            if not quota["ok"]:
                if quota["global_requests_left"] <= 0 or quota["global_tokens_left"] < params["max_tokens"]:
                    reason = "今天全站额度已用完，明天再来吧"
                elif quota["requests_left"] <= 0:
                    reason = "你今天的使用次数已用完，明天再来吧"
                else:
                    reason = f"今天剩余额度不够这次生成（还剩 {quota['tokens_left']} token），把长度调小一点"
                await ws.send_json({"type": "error", "message": reason, "quota": quota})
                continue
            consume_request(ip)

            if RT.sem.locked():
                RT.waiting += 1
                await ws.send_json({"type": "queued", "waiting": RT.waiting, "quota": quota})
                try:
                    await asyncio.wait_for(RT.sem.acquire(), timeout=30)
                except asyncio.TimeoutError:
                    RT.waiting -= 1
                    await ws.send_json({"type": "error", "message": "排队超时，服务器繁忙"})
                    continue
                RT.waiting -= 1
            else:
                await RT.sem.acquire()

            try:
                meta = await _pump(ws, prompt, params)
                if meta:
                    add_tokens(ip, int(meta.get("tokens", 0)))
                    await ws.send_json(
                        {"type": "quota", "quota": quota_status(ip)}
                    )
            finally:
                RT.sem.release()

    except WebSocketDisconnect:
        return
    except RuntimeError:
        # 客户端断开后继续 send_json 会抛 RuntimeError
        return
    finally:
        _conn_count[ip] = max(0, _conn_count.get(ip, 1) - 1)


# ---------------------------------------------------------------- 交流讨论区
DB_PATH = Path(__file__).resolve().parent / "data" / "comments.db"
ADMIN_FILE = Path(__file__).resolve().parent / "admin_password.txt"
SECRET_FILE = Path(__file__).resolve().parent / ".admin_secret"
COOKIE_NAME = "zh_admin"
COOKIE_MAX_AGE = 7 * 24 * 3600
COMMENT_MAX_CHARS = 300
NICK_MAX_CHARS = 16
POST_COOLDOWN = 45.0

_db_conn: Any = None
_db_lock = threading.RLock()  # 可重入：db() 内部与调用方会嵌套加锁
_last_post: dict[str, float] = {}


def _load_admin_password() -> str:
    if ADMIN_FILE.exists():
        value = ADMIN_FILE.read_text(encoding="utf-8").strip()
        if value:
            return value
    pwd = secrets.token_urlsafe(12)
    ADMIN_FILE.write_text(pwd + "\n", encoding="utf-8")
    return pwd


def _load_secret() -> bytes:
    if SECRET_FILE.exists():
        return SECRET_FILE.read_bytes()
    data = secrets.token_bytes(32)
    SECRET_FILE.write_bytes(data)
    return data


ADMIN_PASSWORD = _load_admin_password()
ADMIN_SECRET = _load_secret()


def db() -> Any:
    global _db_conn
    with _db_lock:
        if _db_conn is None:
            DB_PATH.parent.mkdir(parents=True, exist_ok=True)
            _db_conn = sqlite3.connect(DB_PATH, check_same_thread=False)
            _db_conn.row_factory = sqlite3.Row
            _db_conn.execute(
                """
                CREATE TABLE IF NOT EXISTS comments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    nickname TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    ip TEXT NOT NULL DEFAULT ''
                )
                """
            )
            _db_conn.execute(
                """
                CREATE TABLE IF NOT EXISTS usage (
                    day TEXT NOT NULL,
                    ip TEXT NOT NULL,
                    requests INTEGER NOT NULL DEFAULT 0,
                    tokens INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY (day, ip)
                )
                """
            )
            _db_conn.commit()
        return _db_conn


def _today() -> str:
    return time.strftime("%Y-%m-%d")


def usage_of(ip: str, day: str | None = None) -> tuple[int, int]:
    day = day or _today()
    row = db().execute(
        "SELECT requests, tokens FROM usage WHERE day = ? AND ip = ?", (day, ip)
    ).fetchone()
    return (int(row[0]), int(row[1])) if row else (0, 0)


def usage_global(day: str | None = None) -> tuple[int, int]:
    day = day or _today()
    row = db().execute(
        "SELECT COALESCE(SUM(requests),0), COALESCE(SUM(tokens),0) FROM usage WHERE day = ?",
        (day,),
    ).fetchone()
    return int(row[0]), int(row[1])


def consume_request(ip: str) -> None:
    day = _today()
    with _db_lock:
        db().execute(
            """INSERT INTO usage (day, ip, requests, tokens) VALUES (?, ?, 1, 0)
               ON CONFLICT(day, ip) DO UPDATE SET requests = requests + 1""",
            (day, ip),
        )
        db().commit()


def add_tokens(ip: str, tokens: int) -> None:
    if tokens <= 0:
        return
    day = _today()
    with _db_lock:
        db().execute(
            """INSERT INTO usage (day, ip, requests, tokens) VALUES (?, ?, 0, ?)
               ON CONFLICT(day, ip) DO UPDATE SET tokens = tokens + ?""",
            (day, ip, tokens, tokens),
        )
        db().commit()


def quota_status(ip: str, want_tokens: int = 0) -> dict[str, Any]:
    used_req, used_tok = usage_of(ip)
    g_req, g_tok = usage_global()
    req_left = DAILY_REQ_PER_IP - used_req
    tok_left = DAILY_TOKENS_PER_IP - used_tok
    g_req_left = DAILY_REQ_GLOBAL - g_req
    g_tok_left = DAILY_TOKENS_GLOBAL - g_tok
    return {
        "requests_used": used_req,
        "requests_limit": DAILY_REQ_PER_IP,
        "requests_left": max(0, req_left),
        "tokens_used": used_tok,
        "tokens_limit": DAILY_TOKENS_PER_IP,
        "tokens_left": max(0, tok_left),
        "global_requests_left": max(0, g_req_left),
        "global_tokens_left": max(0, g_tok_left),
        "ok": req_left > 0 and tok_left >= max(1, want_tokens) and g_req_left > 0 and g_tok_left >= max(1, want_tokens),
    }


def _client_ip(request: Any) -> str:
    for key in ("x-real-ip", "x-forwarded-for"):
        value = request.headers.get(key)
        if value:
            return value.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _make_token() -> str:
    exp = int(time.time()) + COOKIE_MAX_AGE
    sig = hmac.new(ADMIN_SECRET, str(exp).encode(), hashlib.sha256).hexdigest()
    return f"{exp}.{sig}"


def _is_admin(request: Any) -> bool:
    token = request.cookies.get(COOKIE_NAME)
    if not token or "." not in token:
        return False
    exp_s, sig = token.split(".", 1)
    try:
        if int(exp_s) < time.time():
            return False
    except ValueError:
        return False
    good = hmac.new(ADMIN_SECRET, exp_s.encode(), hashlib.sha256).hexdigest()
    return hmac.compare_digest(sig, good)


def _rows(limit: int) -> list[dict[str, Any]]:
    cur = db().execute(
        "SELECT id, nickname, content, created_at FROM comments ORDER BY id DESC LIMIT ?",
        (limit,),
    )
    return [dict(r) for r in cur.fetchall()]


@app.get("/api/comments")
async def list_comments(limit: int = 100) -> JSONResponse:
    limit = max(1, min(limit, 200))
    total = db().execute("SELECT COUNT(*) FROM comments").fetchone()[0]
    return JSONResponse({"comments": _rows(limit), "total": total})


@app.post("/api/comments")
async def add_comment(request: Request) -> JSONResponse:
    ip = _client_ip(request)
    now = time.time()
    gap = now - _last_post.get(ip, 0.0)
    if gap < POST_COOLDOWN:
        return JSONResponse(
            {"error": f"发得太快了，请等 {int(POST_COOLDOWN - gap) + 1} 秒"},
            status_code=429,
        )

    try:
        payload = await request.json()
    except Exception:
        return JSONResponse({"error": "请求格式错误"}, status_code=400)

    content = str(payload.get("content", "")).strip()
    nickname = str(payload.get("nickname", "")).strip() or "匿名"
    if not content:
        return JSONResponse({"error": "内容不能为空"}, status_code=400)
    if len(content) > COMMENT_MAX_CHARS:
        return JSONResponse({"error": f"最多 {COMMENT_MAX_CHARS} 字"}, status_code=400)
    nickname = nickname[:NICK_MAX_CHARS]

    with _db_lock:
        cur = db().execute(
            "INSERT INTO comments (nickname, content, created_at, ip) VALUES (?, ?, ?, ?)",
            (nickname, content, now, ip),
        )
        db().commit()
        cid = cur.lastrowid
    _last_post[ip] = now
    return JSONResponse(
        {"comment": {"id": cid, "nickname": nickname, "content": content, "created_at": now}},
        status_code=201,
    )


@app.post("/admin/login")
async def admin_login(request: Request) -> JSONResponse:
    try:
        payload = await request.json()
    except Exception:
        return JSONResponse({"error": "请求格式错误"}, status_code=400)
    if not secrets.compare_digest(str(payload.get("password", "")), ADMIN_PASSWORD):
        await asyncio.sleep(0.6)  # 轻量拖慢暴力尝试
        return JSONResponse({"error": "密码错误"}, status_code=401)
    resp = JSONResponse({"ok": True})
    resp.set_cookie(
        COOKIE_NAME, _make_token(), max_age=COOKIE_MAX_AGE,
        httponly=True, samesite="lax", path="/",
    )
    return resp


@app.post("/admin/logout")
async def admin_logout() -> JSONResponse:
    resp = JSONResponse({"ok": True})
    resp.delete_cookie(COOKIE_NAME, path="/")
    return resp


@app.get("/admin/api/session")
async def admin_session(request: Request) -> JSONResponse:
    return JSONResponse({"admin": _is_admin(request)})


@app.get("/admin/api/usage")
async def admin_usage(request: Request) -> JSONResponse:
    if not _is_admin(request):
        return JSONResponse({"error": "未登录"}, status_code=401)
    day = _today()
    cur = db().execute(
        "SELECT ip, requests, tokens FROM usage WHERE day = ? ORDER BY requests DESC, tokens DESC LIMIT 300",
        (day,),
    )
    rows = [dict(r) for r in cur.fetchall()]
    g_req, g_tok = usage_global()
    return JSONResponse(
        {
            "day": day,
            "rows": rows,
            "global": {
                "requests": g_req, "request_limit": DAILY_REQ_GLOBAL,
                "tokens": g_tok, "token_limit": DAILY_TOKENS_GLOBAL,
            },
            "limits": {
                "req_per_ip": DAILY_REQ_PER_IP,
                "tokens_per_ip": DAILY_TOKENS_PER_IP,
            },
        }
    )


@app.delete("/admin/api/usage/{ip}")
async def admin_usage_reset_one(request: Request, ip: str) -> JSONResponse:
    if not _is_admin(request):
        return JSONResponse({"error": "未登录"}, status_code=401)
    with _db_lock:
        db().execute("DELETE FROM usage WHERE day = ? AND ip = ?", (_today(), ip))
        db().commit()
    RT.last_by_ip.pop(ip, None)
    return JSONResponse({"ok": True})


@app.post("/admin/api/usage/reset")
async def admin_usage_reset_all(request: Request) -> JSONResponse:
    if not _is_admin(request):
        return JSONResponse({"error": "未登录"}, status_code=401)
    with _db_lock:
        db().execute("DELETE FROM usage WHERE day = ?", (_today(),))
        db().commit()
    RT.last_by_ip.clear()
    return JSONResponse({"ok": True})


@app.get("/admin/api/comments")
async def admin_comments(request: Request) -> JSONResponse:
    if not _is_admin(request):
        return JSONResponse({"error": "未登录"}, status_code=401)
    cur = db().execute(
        "SELECT id, nickname, content, created_at, ip FROM comments ORDER BY id DESC LIMIT 500"
    )
    return JSONResponse({"comments": [dict(r) for r in cur.fetchall()]})


@app.delete("/admin/api/comments/{cid}")
async def admin_delete(request: Request, cid: int) -> JSONResponse:
    if not _is_admin(request):
        return JSONResponse({"error": "未登录"}, status_code=401)
    with _db_lock:
        cur = db().execute("DELETE FROM comments WHERE id = ?", (cid,))
        db().commit()
        removed = cur.rowcount
    return JSONResponse({"deleted": removed})


@app.get("/admin", response_class=HTMLResponse)
async def admin_page() -> HTMLResponse:
    return HTMLResponse(ADMIN_HTML)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )
    torch.set_num_threads(max(1, args.threads))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    log.info("加载模型……")
    RT.load()
    log.info("监听 http://%s:%d  (ws://%s:%d/ws)", args.host, args.port, args.host, args.port)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning", timeout_keep_alive=75)


if __name__ == "__main__":
    main()
