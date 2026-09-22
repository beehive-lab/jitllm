package org.beehive.jllm.server;

import static org.junit.Assert.*;
import static org.junit.Assume.assumeTrue;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;
import org.beehive.jllm.api.LocalModel;
import org.beehive.jllm.api.LocalModels;
import org.beehive.jllm.api.ModelOptions;
import org.beehive.jllm.golden.GoldenFixture;
import org.beehive.jllm.runtime.backend.BackendId;
import org.beehive.jllm.runtime.policy.ExecutionPolicy;
import org.junit.Test;

/** Real CPU model: HTTP readiness, independent requests, streaming, and clean lifecycle. */
public class ServerLifecycleAccelTest {
    @Test
    public void servesOnSelectedAddressAndPortAndPreservesGenerationLimit() throws Exception {
        var path = GoldenFixture.locate(GoldenFixture.Fixture.LLAMA_3_2_1B_Q8_0);
        assumeTrue("local Llama Q8_0 fixture required", path != null);
        for (var phase : ExecutionPolicy.PhaseStrategy.values()) {
            try (LocalModel model =
                    LocalModels.load(
                            path,
                            ModelOptions.builder()
                                    .contextLength(128)
                                    .backend(BackendId.CPU)
                                    .executionPolicy(
                                            ExecutionPolicy.builder().phaseStrategy(phase).build())
                                    .build())) {
                InferenceService inference = new InferenceService(model);
                try (OpenAIServer server = new OpenAIServer(inference, "fixture", false)) {
                    server.start("127.0.0.1", 0);
                    assertTrue(server.port() > 0);
                    HttpClient client =
                            HttpClient.newBuilder().connectTimeout(Duration.ofSeconds(5)).build();
                    String base = "http://127.0.0.1:" + server.port();
                    assertEquals(
                            200,
                            client.send(
                                            HttpRequest.newBuilder(URI.create(base + "/health"))
                                                    .build(),
                                            HttpResponse.BodyHandlers.ofString())
                                    .statusCode());
                    String body =
                            "{\"messages\":[{\"role\":\"user\",\"content\":\"Hello\"}],\"temperature\":0,\"max_tokens\":2}";
                    HttpRequest request =
                            HttpRequest.newBuilder(URI.create(base + "/v1/chat/completions"))
                                    .timeout(Duration.ofSeconds(60))
                                    .header("Content-Type", "application/json")
                                    .POST(HttpRequest.BodyPublishers.ofString(body))
                                    .build();
                    var response = client.send(request, HttpResponse.BodyHandlers.ofString());
                    assertEquals(response.body(), 200, response.statusCode());
                    var parsed = Json.parseObject(response.body());
                    var usage = (java.util.Map<?, ?>) parsed.get("usage");
                    assertTrue(
                            response.body(),
                            ((Number) usage.get("completion_tokens")).intValue() <= 2);
                    assertTrue(((Number) usage.get("prompt_tokens")).intValue() > 2);
                    var second = client.send(request, HttpResponse.BodyHandlers.ofString());
                    assertEquals(
                            parsed.get("choices"), Json.parseObject(second.body()).get("choices"));
                    String streamBody = body.substring(0, body.length() - 1) + ",\"stream\":true}";
                    var streaming =
                            client.send(
                                    HttpRequest.newBuilder(request.uri())
                                            .timeout(Duration.ofSeconds(60))
                                            .POST(HttpRequest.BodyPublishers.ofString(streamBody))
                                            .build(),
                                    HttpResponse.BodyHandlers.ofString());
                    assertEquals(200, streaming.statusCode());
                    assertTrue(streaming.body().contains("data: [DONE]"));
                }
                // Closing the model here would fail if server.close() left its session open.
            }
        }
    }
}
