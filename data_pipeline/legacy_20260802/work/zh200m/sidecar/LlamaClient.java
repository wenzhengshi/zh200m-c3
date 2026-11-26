import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;
import java.util.Scanner;

/**
 * llama.cpp /completion SSE 客户端示例（Spring Boot 环境建议使用 WebClient/Reactor 封装）。
 * 会话 KV Cache：保持同一 slot（llama-server slot 默认复用），completion 请求带
 * "cache_prompt": true，并用 "slot_id" 固定会话；取消通过中断 SSE 读取实现。
 */
public class LlamaClient {
    record Completion(String prompt, String sessionId) {}

    public static void main(String[] args) throws Exception {
        String base = System.getenv().getOrDefault("LLAMACPP_BASE", "http://127.0.0.1:8080");
        String model = System.getenv().getOrDefault("LLAMACPP_MODEL", "q5_k_m");
        var client = HttpClient.newBuilder()
                .connectTimeout(Duration.ofSeconds(10))
                .build();
        var body = """
                {
                  "prompt": "新华社报道，",
                  "n_predict": 64,
                  "temperature": 0.2,
                  "cache_prompt": true,
                  "slot_id": -1,
                  "stream": true
                }
                """;
        var req = HttpRequest.newBuilder(URI.create(base + "/completion"))
                .timeout(Duration.ofSeconds(120))
                .header("Content-Type", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString(body))
                .build();
        try (var resp = client.send(req, HttpResponse.BodyHandlers.ofInputStream())) {
            if (resp.statusCode() != 200) {
                throw new IllegalStateException("HTTP " + resp.statusCode());
            }
            // SSE 流式读取：逐行解析 "data: {...}"，遇到中断即取消（关闭流）
            try (Scanner sc = new Scanner(resp.body())) {
                while (sc.hasNextLine()) {
                    String line = sc.nextLine();
                    if (line.startsWith("data: ")) {
                        System.out.print(line.substring(6));
                        System.out.flush();
                    }
                }
            }
        }
    }
}
