# llama.cpp CPU 推理 Sidecar（Spring Boot 集成方案）

## 组成

- `start_cpu_server.sh`：Linux 启动脚本（线程/上下文/端口可配）
- `start_cpu_server.bat`：Windows 启动脚本
- `application.yml`：Spring Boot 配置示例
- `LlamaClient.java`：Java 流式客户端（SSE，超时与取消）

## 设计要点

1. Spring Boot 通过 `http://127.0.0.1:8080` 调用本地 llama-server（`/completion`、`/health`、`/props`）。
2. 流式输出：SSE `data:` 逐块；取消 = 关闭响应流（llama-server 停止生成）。
3. 超时：HttpClient 120s 连接/请求超时；服务端 n_predict 限制单次生成量。
4. 会话 KV Cache：`cache_prompt=true` 复用已计算的 prompt KV；固定 `slot_id` 保持会话；
   会话结束调用 `/slots` 释放或切换 slot。
5. 线程数：`-t` 配置（application.yml 中 `llamacpp.threads`）。
6. 模型切换：服务端启动时 `-m` 指定；运行期通过配置切换重启，或按变体启动多实例端口。

## 推荐模型变体

- `model-q5_k_m.gguf`：质量优先（默认推荐）
- `model-q4_k_m.gguf`：速度/内存优先
- `model-q8_0.gguf`：高质量对照
