package org.beehive.jllm.server;

import static org.beehive.jllm.model.loader.ModelLoader.loadModel;

import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpServer;
import java.io.IOException;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicLong;
import org.beehive.jllm.api.ChatMessage;
import org.beehive.jllm.api.ChatRole;
import org.beehive.jllm.api.LocalModel;
import org.beehive.jllm.api.LocalModels;
import org.beehive.jllm.api.ModelOptions;
import org.beehive.jllm.integration.cli.StartupDiagnostics;
import org.beehive.jllm.model.Model;

/**
 * OpenAI-compatible HTTP server for jllm, built on the JDK {@link HttpServer} (no external
 * dependencies). Exposes the loaded model behind the endpoints an OpenAI client already speaks:
 *
 * <ul>
 *   <li>{@code POST /v1/chat/completions} — chat, streaming (SSE) or full JSON.
 *   <li>{@code POST /v1/completions} — text completion (prompt as a single user turn).
 *   <li>{@code GET /v1/models} — the one served model.
 *   <li>{@code GET /health} — liveness.
 * </ul>
 *
 * <p>Generation is serialized on a single {@link InferenceService} (the GPU is one context); HTTP
 * accept is multi-threaded so clients queue cleanly. Run:
 *
 * <pre>
 *   java. org.beehive.jllm.server.OpenAIServer --model model.gguf --port 8080 --gpu
 * </pre>
 */
public final class OpenAIServer implements AutoCloseable {

    /**
     * Either path, behind one call. The engine-backed service batches several conversations; the
     * original serializes them behind a lock. The HTTP handlers do not need to know which.
     */
    private interface Generator extends AutoCloseable {
        InferenceService.Result generate(
                InferenceService.Request request, java.util.function.Consumer<String> onToken);

        default boolean greedyOnly() {
            return false;
        }

        @Override
        void close();
    }

    private final Generator service;
    private final String servedModel;
    private final boolean gpu;
    private int port;
    private HttpServer http;
    private java.util.concurrent.ExecutorService httpWorkers;
    private LocalModel ownedModel;
    private boolean closed;
    private final AtomicLong seq = new AtomicLong();

    public OpenAIServer(InferenceService service, String servedModel, boolean gpu) {
        this(wrap(service), servedModel, gpu);
    }

    /** Engine-backed: concurrent requests share one batch instead of one lock. */
    public OpenAIServer(EngineInferenceService service, String servedModel, boolean gpu) {
        this(wrap(service), servedModel, gpu);
    }

    private static Generator wrap(InferenceService delegate) {
        return new Generator() {
            @Override
            public InferenceService.Result generate(
                    InferenceService.Request request, java.util.function.Consumer<String> onToken) {
                return delegate.generate(request, onToken);
            }

            @Override
            public void close() {
                delegate.close();
            }
        };
    }

    private static Generator wrap(EngineInferenceService delegate) {
        return new Generator() {
            @Override
            public InferenceService.Result generate(
                    InferenceService.Request request, java.util.function.Consumer<String> onToken) {
                return delegate.generate(request, onToken);
            }

            @Override
            public boolean greedyOnly() {
                return true;
            }

            @Override
            public void close() {
                delegate.close();
            }
        };
    }

    private OpenAIServer(Generator service, String servedModel, boolean gpu) {
        this.service = service;
        this.servedModel = servedModel;
        this.gpu = gpu;
    }

    public static void main(String[] args) throws IOException {
        if (java.util.Arrays.asList(args).contains("--help")
                || java.util.Arrays.asList(args).contains("-h")) {
            System.out.println(
                    "Usage: OpenAIServer [serve] --model FILE [--ctx-size N] [--host 127.0.0.1] [--port 8080] [--gpu] [--parallel N] [-v]");
            return;
        }
        ServerOptions options = ServerOptions.parse(args);
        var config = options.model();
        // HTTP requests choose sampling independently; a greedy-only device sampler cannot be
        // pinned globally.
        System.clearProperty("jllm.deviceSample");
        long startedNs = System.nanoTime();
        Path path = config.model();
        String served = path.getFileName().toString().replaceAll("\\.gguf$", "");
        ModelOptions modelOptions = config.modelOptions();
        OpenAIServer server;
        if (options.parallel() > 1) {
            Model model = loadModel(path, config.contextLength(), true, config.gpu());
            long loadNs = System.nanoTime() - startedNs;
            if (model.weights().dataType() != org.beehive.jllm.runtime.tensor.DataType.F16) {
                throw new IllegalArgumentException(
                        "Parallel serving requires FP16 Llama/Qwen3 weights");
            }
            EngineInferenceService service =
                    new EngineInferenceService(
                            model,
                            options.parallel(),
                            options.maxQueuedRequests(),
                            config.contextLength(),
                            options.prefixCacheEntries());
            server = new OpenAIServer(service, served, config.gpu());
            try {
                if (StartupDiagnostics.verbose()) {
                    System.err.print(
                            StartupDiagnostics.render(
                                    model,
                                    path,
                                    service.executionInfo(),
                                    "greedy (parallel serving)",
                                    modelOptions,
                                    loadNs,
                                    startedNs));
                }
                server.start(options.host(), options.port());
            } catch (IOException | RuntimeException | Error failure) {
                server.close();
                throw failure;
            }
        } else {
            LocalModel model = LocalModels.load(path, modelOptions);
            long loadNs = System.nanoTime() - startedNs;
            InferenceService service;
            try {
                service = new InferenceService(model);
            } catch (RuntimeException | Error failure) {
                model.close();
                throw failure;
            }
            server = new OpenAIServer(service, served, config.gpu());
            server.ownedModel = model;
            try {
                if (StartupDiagnostics.verbose()) {
                    System.err.print(
                            StartupDiagnostics.render(
                                    model,
                                    service.prepare(),
                                    "per HTTP request",
                                    modelOptions,
                                    loadNs,
                                    startedNs));
                }
                server.start(options.host(), options.port());
            } catch (IOException | RuntimeException | Error failure) {
                server.close();
                throw failure;
            }
        }
        if (StartupDiagnostics.verbose()) {
            System.err.println(
                    "[server] context per request="
                            + config.contextLength()
                            + " parallel="
                            + options.parallel()
                            + " maxQueuedRequests="
                            + options.maxQueuedRequests());
        }
        Runtime.getRuntime().addShutdownHook(new Thread(server::close, "server-shutdown"));
    }

    public void start(int port) throws IOException {
        start("127.0.0.1", port);
    }

    public synchronized void start(String host, int port) throws IOException {
        if (http != null || closed)
            throw new IllegalStateException("Server already started or closed");
        http = HttpServer.create(new InetSocketAddress(host, port), 0);
        this.port = http.getAddress().getPort();
        http.createContext("/", this::handleIndex);
        http.createContext("/health", this::handleHealth);
        http.createContext("/v1/models", this::handleModels);
        http.createContext("/v1/chat/completions", ex -> handleCompletion(ex, true));
        http.createContext("/v1/completions", ex -> handleCompletion(ex, false));
        httpWorkers = Executors.newFixedThreadPool(8);
        http.setExecutor(httpWorkers);
        http.start();
        String bound = http.getAddress().getAddress().getHostAddress();
        if (bound.contains(":")) bound = "[" + bound + "]";
        System.err.println(
                "[server] listening on http://"
                        + bound
                        + ":"
                        + this.port
                        + "  model="
                        + servedModel);
    }

    public synchronized int port() {
        return port;
    }

    @Override
    public synchronized void close() {
        if (closed) return;
        closed = true;
        if (http != null) http.stop(1);
        if (httpWorkers != null) httpWorkers.shutdownNow();
        try {
            service.close();
        } finally {
            if (ownedModel != null) ownedModel.close();
        }
    }

    // ── Endpoints ─────────────────────────────────────────────────────────────

    private void handleIndex(HttpExchange ex) throws IOException {
        if (!"/".equals(ex.getRequestURI().getPath())) {
            sendError(ex, 404, "Not found");
            return;
        }
        byte[] bytes =
                INDEX_HTML
                        .replace("{{model}}", servedModel)
                        .replace("{{backend}}", gpu ? "GPU (TornadoVM)" : "CPU")
                        .replace("{{port}}", String.valueOf(port))
                        .getBytes(StandardCharsets.UTF_8);
        ex.getResponseHeaders().set("Content-Type", "text/html; charset=utf-8");
        ex.sendResponseHeaders(200, bytes.length);
        try (OutputStream os = ex.getResponseBody()) {
            os.write(bytes);
        }
    }

    private static final String INDEX_HTML =
            """
            <!doctype html>
            <html lang="en">
            <head>
            <meta charset="utf-8">
            <title>jllm server</title>
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <style>
              body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
                     max-width: 780px; margin: 40px auto; padding: 0 20px; line-height: 1.5; color: #1a1a1a; }
              h1 { font-size: 1.5rem; margin-bottom: 0; }
              .sub { color: #666; margin-top: 4px; }
              .badges span { display: inline-block; background: #eee; border-radius: 4px; padding: 2px 8px;
                             margin: 4px 6px 0 0; font-size: 0.85rem; }
              table { border-collapse: collapse; margin: 12px 0; }
              td { padding: 3px 10px 3px 0; vertical-align: top; }
              td.k { color: #666; }
              code, pre { background: #f4f4f4; border-radius: 4px; }
              code { padding: 1px 5px; }
              pre { padding: 10px; overflow-x: auto; }
              a { color: #0969da; }
              hr { border: none; border-top: 1px solid #eee; margin: 24px 0; }
              ul { padding-left: 20px; }
            </style>
            </head>
            <body>
              <h1>jllm</h1>
              <div class="sub">Local OpenAI-compatible inference server</div>

              <table>
                <tr><td class="k">Model</td><td><code>{{model}}</code></td></tr>
                <tr><td class="k">Backend</td><td>{{backend}}</td></tr>
                <tr><td class="k">Port</td><td>{{port}}</td></tr>
              </table>

              <p>
                This is a local instance of
                <a href="https://github.com/beehive-lab/jllm" target="_blank">jllm</a>,
                a Llama3-family inference engine written in native Java and automatically accelerated on
                GPUs with <a href="https://github.com/beehive-lab/TornadoVM" target="_blank">TornadoVM</a>.
                It supports Llama3, Mistral, Devstral 2, Qwen2.5, Qwen3, Phi-3, IBM Granite 3.2+, and
                IBM Granite 4.0 models in GGUF format, and is also used as the GPU inference engine behind
                the <a href="https://docs.quarkiverse.io/quarkus-langchain4j/dev/gpullama3-chat-model.html" target="_blank">Quarkus</a>
                and <a href="https://docs.langchain4j.dev/integrations/language-models/gpullama3-java" target="_blank">LangChain4j</a>
                integrations.
              </p>

              <hr>

              <h3>API endpoints</h3>
              <ul>
                <li><code>GET  /health</code> — liveness check</li>
                <li><code>GET  /v1/models</code> — list the served model</li>
                <li><code>POST /v1/chat/completions</code> — chat completions (supports <code>"stream": true</code> SSE)</li>
                <li><code>POST /v1/completions</code> — text completions (supports <code>"stream": true</code> SSE)</li>
              </ul>

              <h3>Quick test</h3>
              <pre><code>curl http://localhost:{{port}}/v1/chat/completions \\
      -H "Content-Type: application/json" \\
      -d '{
            "model": "{{model}}",
            "messages": [{"role": "user", "content": "Hello!"}]
          }'</code></pre>

              <p class="sub">
                Any OpenAI-compatible client (e.g. the <code>openai</code> Python/Node SDK) can point its
                base URL at this server. Generation runs on a single serialized GPU/CPU context, so
                concurrent requests are queued and processed one at a time.
              </p>
            </body>
            </html>
            """;

    private void handleHealth(HttpExchange ex) throws IOException {
        sendJson(ex, 200, Map.of("status", "ok"));
    }

    private void handleModels(HttpExchange ex) throws IOException {
        Map<String, Object> entry = new LinkedHashMap<>();
        entry.put("id", servedModel);
        entry.put("object", "model");
        entry.put("created", 0);
        entry.put("owned_by", "jllm");
        sendJson(ex, 200, Map.of("object", "list", "data", List.of(entry)));
    }

    @SuppressWarnings("unchecked")
    private void handleCompletion(HttpExchange ex, boolean chat) throws IOException {
        if (!"POST".equals(ex.getRequestMethod())) {
            sendError(ex, 405, "Method not allowed — use POST");
            return;
        }
        Map<String, Object> body;
        try {
            String raw = new String(ex.getRequestBody().readAllBytes(), StandardCharsets.UTF_8);
            body = Json.parseObject(raw);
        } catch (Exception e) {
            sendError(ex, 400, "Invalid JSON body: " + e.getMessage());
            return;
        }

        List<ChatMessage> messages = new ArrayList<>();
        try {
            if (chat) {
                Object msgs = body.get("messages");
                if (!(msgs instanceof List) || ((List<?>) msgs).isEmpty()) {
                    sendError(ex, 400, "'messages' must be a non-empty array");
                    return;
                }
                for (Object o : (List<Object>) msgs) {
                    Map<String, Object> m = (Map<String, Object>) o;
                    String role = Json.str(m, "role", "user");
                    String content = Json.str(m, "content", "");
                    messages.add(ChatMessage.of(chatRole(role), content));
                }
            } else {
                Object prompt = body.get("prompt");
                String text =
                        prompt instanceof String s ? s : prompt == null ? "" : prompt.toString();
                if (text.isEmpty()) {
                    sendError(ex, 400, "'prompt' must be a non-empty string");
                    return;
                }
                messages.add(ChatMessage.of(ChatRole.USER, text));
            }
        } catch (Exception e) {
            sendError(ex, 400, "Malformed request: " + e.getMessage());
            return;
        }

        int maxTokens =
                Json.intVal(
                        body,
                        "max_tokens",
                        chat ? Json.intVal(body, "max_completion_tokens", 256) : 256);
        float temperature = (float) Json.num(body, "temperature", 0.0);
        float topP = (float) Json.num(body, "top_p", 0.95);
        long seed = (long) Json.num(body, "seed", 1234);
        boolean stream = Json.bool(body, "stream", false);
        if (service.greedyOnly() && temperature != 0.0f) {
            sendError(
                    ex, 400, "Parallel serving currently requires temperature=0 (greedy sampling)");
            return;
        }

        var req = new InferenceService.Request(messages, maxTokens, temperature, topP, seed);
        String id = (chat ? "chatcmpl-" : "cmpl-") + seq.incrementAndGet();
        long created = System.currentTimeMillis() / 1000;

        if (stream) {
            streamResponse(ex, req, id, created, chat);
        } else {
            fullResponse(ex, req, id, created, chat);
        }
    }

    /**
     * The OpenAI role names, mapped onto the facade's roles. An unknown role is a client error, not
     * a silent downgrade to "user": a request whose role the server does not understand would be
     * rendered into the wrong place in the chat template.
     */
    private static ChatRole chatRole(String role) {
        return switch (role) {
            case "system" -> ChatRole.SYSTEM;
            case "user" -> ChatRole.USER;
            case "assistant" -> ChatRole.ASSISTANT;
            case "tool" -> ChatRole.TOOL;
            default -> throw new IllegalArgumentException("unknown role: " + role);
        };
    }

    // ── Non-streaming ─────────────────────────────────────────────────────────

    private void fullResponse(
            HttpExchange ex, InferenceService.Request req, String id, long created, boolean chat)
            throws IOException {
        InferenceService.Result r;
        try {
            r = service.generate(req, null);
        } catch (Exception e) {
            sendError(ex, 500, "Generation failed: " + e);
            return;
        }
        String finish = r.stopped() ? "stop" : "length";
        Map<String, Object> choice = new LinkedHashMap<>();
        choice.put("index", 0);
        if (chat) {
            choice.put("message", Map.of("role", "assistant", "content", r.text()));
        } else {
            choice.put("text", r.text());
        }
        choice.put("finish_reason", finish);

        Map<String, Object> resp = new LinkedHashMap<>();
        resp.put("id", id);
        resp.put("object", chat ? "chat.completion" : "text_completion");
        resp.put("created", created);
        resp.put("model", servedModel);
        resp.put("choices", List.of(choice));
        resp.put(
                "usage",
                Map.of(
                        "prompt_tokens", r.promptTokens(),
                        "completion_tokens", r.completionTokens(),
                        "total_tokens", r.promptTokens() + r.completionTokens()));
        sendJson(ex, 200, resp);
    }

    // ── Streaming (Server-Sent Events) ────────────────────────────────────────

    private void streamResponse(
            HttpExchange ex, InferenceService.Request req, String id, long created, boolean chat)
            throws IOException {
        ex.getResponseHeaders().set("Content-Type", "text/event-stream; charset=utf-8");
        ex.getResponseHeaders().set("Cache-Control", "no-cache");
        ex.getResponseHeaders().set("Connection", "keep-alive");
        ex.sendResponseHeaders(200, 0);
        OutputStream os = ex.getResponseBody();

        String object = chat ? "chat.completion.chunk" : "text_completion";
        try {
            if (chat) {
                // First chunk carries the assistant role.
                writeSse(os, chunk(id, object, created, roleDelta(), null));
            }
            InferenceService.Result r =
                    service.generate(
                            req,
                            piece -> {
                                try {
                                    writeSse(
                                            os,
                                            chunk(
                                                    id,
                                                    object,
                                                    created,
                                                    chat ? contentDelta(piece) : textField(piece),
                                                    null));
                                } catch (IOException io) {
                                    throw new RuntimeException(
                                            io); // client disconnected — abort generation
                                }
                            });
            String finish = r.stopped() ? "stop" : "length";
            writeSse(os, chunk(id, object, created, chat ? Map.of() : textField(""), finish));
            os.write("data: [DONE]\n\n".getBytes(StandardCharsets.UTF_8));
            os.flush();
        } catch (Exception e) {
            // Best-effort: nothing more we can send on a half-written SSE stream.
        } finally {
            os.close();
        }
    }

    private Map<String, Object> roleDelta() {
        Map<String, Object> d = new LinkedHashMap<>();
        d.put("role", "assistant");
        d.put("content", "");
        return d;
    }

    private Map<String, Object> contentDelta(String piece) {
        return Map.of("content", piece);
    }

    private Map<String, Object> textField(String piece) {
        return Map.of("text", piece);
    }

    /** One SSE data object. {@code delta} is the chat delta map or the completion text map. */
    private String chunk(
            String id, String object, long created, Map<String, Object> delta, String finish) {
        Map<String, Object> choice = new LinkedHashMap<>();
        choice.put("index", 0);
        if (object.startsWith("chat")) {
            choice.put("delta", delta);
        } else {
            choice.putAll(delta); // {"text": ...}
        }
        choice.put("finish_reason", finish);
        Map<String, Object> obj = new LinkedHashMap<>();
        obj.put("id", id);
        obj.put("object", object);
        obj.put("created", created);
        obj.put("model", servedModel);
        obj.put("choices", List.of(choice));
        return Json.write(obj);
    }

    private void writeSse(OutputStream os, String jsonData) throws IOException {
        os.write(("data: " + jsonData + "\n\n").getBytes(StandardCharsets.UTF_8));
        os.flush();
    }

    // ── Wire helpers ──────────────────────────────────────────────────────────

    private void sendJson(HttpExchange ex, int status, Map<String, Object> body)
            throws IOException {
        byte[] bytes = Json.write(body).getBytes(StandardCharsets.UTF_8);
        ex.getResponseHeaders().set("Content-Type", "application/json; charset=utf-8");
        ex.sendResponseHeaders(status, bytes.length);
        try (OutputStream os = ex.getResponseBody()) {
            os.write(bytes);
        }
    }

    private void sendError(HttpExchange ex, int status, String message) throws IOException {
        Map<String, Object> err =
                Map.of("error", Map.of("message", message, "type", "invalid_request_error"));
        sendJson(ex, status, err);
    }
}
