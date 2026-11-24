import asyncio
import json
import sys

import websockets


async def one(url: str, prompt: str, max_tokens: int, temperature: float) -> None:
    async with websockets.connect(url, max_size=2**20) as ws:
        await ws.send(json.dumps({
            "type": "generate",
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_k": 40,
            "top_p": 0.9,
            "repetition_penalty": 1.2,
        }))
        text = ""
        async for raw in ws:
            msg = json.loads(raw)
            kind = msg["type"]
            if kind == "start":
                print(f"提示: {prompt}")
                print("续写: ", end="", flush=True)
            elif kind == "token":
                text += msg["text"]
                print(msg["text"], end="", flush=True)
            elif kind == "queued":
                print(f"[排队 {msg['waiting']}]", flush=True)
            elif kind == "done":
                print()
                print(f"  -> {msg['tokens']} tok, ttft={msg['ttft']}s, "
                      f"elapsed={msg['elapsed']}s, {msg['tok_per_s']} tok/s")
                break
            elif kind == "error":
                print(f"\n  !! error: {msg['message']}")
                break


async def main() -> None:
    url = sys.argv[1] if len(sys.argv) > 1 else "ws://127.0.0.1:8765/ws"
    for prompt in ["在很久很久以前，", "春眠不觉晓，", "记者昨日从市气象台获悉，"]:
        await one(url, prompt, 48, 0.9)
        print()


asyncio.run(main())
