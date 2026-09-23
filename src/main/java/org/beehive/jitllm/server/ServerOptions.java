package org.beehive.jitllm.server;

import java.nio.file.Path;
import org.beehive.jitllm.integration.cli.ModelRunConfig;

/** HTTP configuration, independent of terminal prompts and sampling. */
record ServerOptions(
        ModelRunConfig model,
        String host,
        int port,
        int batchSlots,
        int maxQueuedRequests,
        int prefixCacheEntries) {
    // The checks use the parameter, not continuousBatching(): in a compact constructor the
    // fields are not assigned until it returns, so the accessor would read the default.
    ServerOptions {
        if (host.isBlank() || port < 0 || port > 65535)
            throw new IllegalArgumentException("Expected a nonempty --host and --port in 0..65535");
        if (batchSlots < 1 || maxQueuedRequests < 1 || prefixCacheEntries < 0)
            throw new IllegalArgumentException(
                    "Invalid server concurrency or queue/cache capacity");
        if (batchSlots > 1 && !model.gpu())
            throw new IllegalArgumentException("--continuous-batching requires --gpu");
        if (batchSlots <= 1 && prefixCacheEntries > 0)
            throw new IllegalArgumentException(
                    "--prefix-cache-entries requires --continuous-batching");
        if (batchSlots > 1
                && (Boolean.getBoolean("jitllm.withPrefillDecode")
                        || Integer.getInteger("jitllm.prefillBatchSize", 1) > 1
                        || Boolean.getBoolean("jitllm.cudaGraphs"))) {
            throw new IllegalArgumentException(
                    "Continuous batching does not support prefill chunking or CUDA graphs");
        }
        if (batchSlots > 1
                && Boolean.getBoolean(
                        org.beehive.jitllm.runtime.policy.ExecutionPolicy
                                .NATIVE_LIBRARIES_PROPERTY)) {
            throw new IllegalArgumentException(
                    "Continuous batching has no native-library implementation; drop"
                            + " --with-native-libraries");
        }
    }

    /** Whether requests are decoded together by the experimental continuous-batch engine. */
    boolean continuousBatching() {
        return batchSlots > 1;
    }

    /**
     * The warning printed when continuous batching is enabled: it is experimental, and narrower
     * than the regular path.
     */
    static final String CONTINUOUS_BATCHING_WARNING =
            "WARNING: continuous batching is experimental. It currently supports CUDA tensor-core"
                    + " devices, FP16 Llama/Qwen3 weights and greedy requests (temperature 0);"
                    + " prefill chunking and"
                    + " CUDA graphs are not supported, and its setup time and memory are not"
                    + " reported.";

    static ServerOptions parse(String[] args) {
        Path path = null;
        String host = "127.0.0.1";
        // 0 serves the model's own context, as ModelOptions and the loaders define it.
        int port = 8080, context = 0, batchSlots = 1;
        int queued = Integer.getInteger("server.maxQueuedRequests", 64);
        int prefixes = Integer.getInteger("server.prefixCacheEntries", 0);
        boolean gpu = Boolean.getBoolean("use.tornadovm"), contextSet = false;
        for (int i = 0; i < args.length; i++) {
            String option = args[i];
            if (i == 0 && option.equals("serve")) continue;
            switch (option) {
                case "--gpu" -> {
                    gpu = true;
                    continue;
                }
                case "--verbose", "-v" -> {
                    System.setProperty("jitllm.verbose", "true");
                    continue;
                }
                case "--fp32-kv-cache" -> {
                    System.setProperty(
                            org.beehive.jitllm.runtime.policy.StorageOptions.FP32_PROPERTY, "true");
                    continue;
                }
                case "--print-taskgraph-chain" -> {
                    System.setProperty(
                            org.beehive.jitllm.backend.tornado.TaskGraphChainPrinter.PROPERTY,
                            "true");
                    continue;
                }
                case "--with-native-libraries" -> {
                    System.setProperty(
                            org.beehive.jitllm.runtime.policy.ExecutionPolicy
                                    .NATIVE_LIBRARIES_PROPERTY,
                            "true");
                    continue;
                }
                case "--fp16-kv-cache" ->
                        throw new IllegalArgumentException(
                                "--fp16-kv-cache was removed: FP16 is now the default key/value"
                                        + " cache. Drop the flag, or pass --fp32-kv-cache to keep"
                                        + " an FP32 cache");
                case "--cuda-graphs" -> {
                    System.setProperty("jitllm.cudaGraphs", "true");
                    continue;
                }
                case "--with-prefill-decode" -> {
                    System.setProperty("jitllm.withPrefillDecode", "true");
                    continue;
                }
                default -> {}
            }
            String value;
            if (option.contains("=")) {
                String[] pair = option.split("=", 2);
                option = pair[0];
                value = pair[1];
            } else {
                if (++i >= args.length)
                    throw new IllegalArgumentException("Missing value for " + option);
                value = args[i];
            }
            switch (option) {
                case "--model", "-m" -> path = Path.of(value);
                case "--host" -> host = value;
                case "--port", "-p" -> port = Integer.parseInt(value);
                case "--ctx-size", "-c", "--ctx", "--context-length", "--max-tokens", "-n" -> {
                    if (contextSet)
                        throw new IllegalArgumentException("Specify context capacity once");
                    contextSet = true;
                    context = Integer.parseInt(value);
                    if (option.equals("--max-tokens") || option.equals("-n"))
                        System.err.println("--max-tokens/-n is deprecated; use --ctx-size/-c.");
                }
                case "--continuous-batching", "--batch", "-b" -> {
                    batchSlots = Integer.parseInt(value);
                    if (batchSlots < 2)
                        throw new IllegalArgumentException(
                                option + " needs at least 2 request slots");
                    if (!option.equals("--continuous-batching"))
                        System.err.println(option + " is deprecated; use --continuous-batching.");
                }
                case "--max-queued-requests" -> queued = Integer.parseInt(value);
                case "--prefix-cache-entries" -> prefixes = Integer.parseInt(value);
                case "--batch-prefill-size" -> {
                    int width = Integer.parseInt(value);
                    if (width < 1)
                        throw new IllegalArgumentException("--batch-prefill-size must be positive");
                    System.setProperty("jitllm.prefillBatchSize", value);
                    System.setProperty("jitllm.withPrefillDecode", "true");
                }
                default -> throw new IllegalArgumentException("Unknown server option: " + option);
            }
        }
        if (path == null) throw new IllegalArgumentException("serve requires --model");
        return new ServerOptions(
                new ModelRunConfig(path, context, gpu), host, port, batchSlots, queued, prefixes);
    }
}
