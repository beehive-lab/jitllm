package org.beehive.jllm.server;

import java.nio.file.Path;
import org.beehive.jllm.integration.cli.ModelRunConfig;

/** HTTP configuration, independent of terminal prompts and sampling. */
record ServerOptions(
        ModelRunConfig model,
        String host,
        int port,
        int parallel,
        int maxQueuedRequests,
        int prefixCacheEntries) {
    ServerOptions {
        if (host.isBlank() || port < 0 || port > 65535)
            throw new IllegalArgumentException("Expected a nonempty --host and --port in 0..65535");
        if (parallel < 1 || maxQueuedRequests < 1 || prefixCacheEntries < 0)
            throw new IllegalArgumentException(
                    "Invalid server concurrency or queue/cache capacity");
        if (parallel > 1 && !model.gpu())
            throw new IllegalArgumentException("--parallel > 1 requires --gpu");
        if (parallel == 1 && prefixCacheEntries > 0)
            throw new IllegalArgumentException("--prefix-cache-entries requires --parallel > 1");
        if (parallel > 1
                && (Boolean.getBoolean("jllm.withPrefillDecode")
                        || Integer.getInteger("jllm.prefillBatchSize", 1) > 1
                        || Boolean.getBoolean("jllm.cudaGraphs")
                        || Boolean.getBoolean("jllm.kvcache.fp16"))) {
            throw new IllegalArgumentException(
                    "Parallel serving requires FP32 KV cache and does not support prefill chunking or CUDA graphs");
        }
    }

    static ServerOptions parse(String[] args) {
        Path path = null;
        String host = "127.0.0.1";
        // 0 serves the model's own context, as ModelOptions and the loaders define it.
        int port = 8080, context = 0, parallel = 1;
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
                    System.setProperty("jllm.verbose", "true");
                    continue;
                }
                case "--fp16-kv-cache" -> {
                    System.setProperty("jllm.kvcache.fp16", "true");
                    continue;
                }
                case "--cuda-graphs" -> {
                    System.setProperty("jllm.cudaGraphs", "true");
                    continue;
                }
                case "--with-prefill-decode" -> {
                    System.setProperty("jllm.withPrefillDecode", "true");
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
                case "--parallel", "--batch", "-b" -> {
                    parallel = Integer.parseInt(value);
                    if (!option.equals("--parallel"))
                        System.err.println("--batch/-b is deprecated; use --parallel.");
                }
                case "--max-queued-requests" -> queued = Integer.parseInt(value);
                case "--prefix-cache-entries" -> prefixes = Integer.parseInt(value);
                case "--batch-prefill-size" -> {
                    int width = Integer.parseInt(value);
                    if (width < 1)
                        throw new IllegalArgumentException("--batch-prefill-size must be positive");
                    System.setProperty("jllm.prefillBatchSize", value);
                    System.setProperty("jllm.withPrefillDecode", "true");
                }
                default -> throw new IllegalArgumentException("Unknown server option: " + option);
            }
        }
        if (path == null) throw new IllegalArgumentException("serve requires --model");
        return new ServerOptions(
                new ModelRunConfig(path, context, gpu), host, port, parallel, queued, prefixes);
    }
}
