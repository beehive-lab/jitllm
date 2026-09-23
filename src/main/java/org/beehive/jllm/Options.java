package org.beehive.jllm;

import java.io.PrintStream;
import java.nio.file.Path;
import java.nio.file.Paths;

public record Options(
        Path modelPath,
        String prompt,
        String systemPrompt,
        String suffix,
        boolean interactive,
        float temperature,
        float topp,
        long seed,
        int maxTokens,
        boolean stream,
        boolean echo,
        boolean useTornadovm,
        boolean withPrefillDecode,
        int batchPrefillSize,
        int maxNewTokens) {

    /** Compatibility constructor: the former limit sized context and bounded generation. */
    public Options(
            Path modelPath,
            String prompt,
            String systemPrompt,
            String suffix,
            boolean interactive,
            float temperature,
            float topp,
            long seed,
            int maxTokens,
            boolean stream,
            boolean echo,
            boolean useTornadovm,
            boolean withPrefillDecode,
            int batchPrefillSize) {
        this(
                modelPath,
                prompt,
                systemPrompt,
                suffix,
                interactive,
                temperature,
                topp,
                seed,
                maxTokens,
                stream,
                echo,
                useTornadovm,
                withPrefillDecode,
                batchPrefillSize,
                maxTokens);
    }

    public int contextLength() {
        return maxTokens;
    }

    public static final int DEFAULT_MAX_TOKENS = 1024;

    public Options {
        require(maxTokens > 0, "--ctx-size must be positive");
        require(maxNewTokens > 0, "--max-new-tokens must be positive");
        require(
                interactive || prompt != null,
                "Missing argument: --prompt is required in --instruct mode e.g. --prompt \"Why is the sky blue?\"");
        require(
                Float.isNaN(temperature) || 0 <= temperature,
                "Invalid argument: --temperature must be non-negative");
        require(
                Float.isNaN(topp) || 0 <= topp && topp <= 1,
                "Invalid argument: --top-p must be within [0, 1]");
        require(batchPrefillSize >= 1, "Invalid argument: --batch-prefill-size must be >= 1");
        require(
                batchPrefillSize == 1 || withPrefillDecode,
                "Invalid argument: --batch-prefill-size requires --with-prefill-decode");
        // Publish to system properties so TornadoVMMasterPlan and Llama read the right values
        // even when the JAR is invoked directly (without the Python launcher).
        if (withPrefillDecode) System.setProperty("jllm.withPrefillDecode", "true");
        if (batchPrefillSize > 1)
            System.setProperty("jllm.prefillBatchSize", String.valueOf(batchPrefillSize));
    }

    static void require(boolean condition, String messageFormat, Object... args) {
        if (!condition) {
            throw new IllegalArgumentException(messageFormat.formatted(args));
        }
    }

    private static boolean getDefaultTornadoVM() {
        return Boolean.parseBoolean(System.getProperty("use.tornadovm", "false"));
    }

    public static void printUsage(PrintStream out) {
        out.println("Usage: jllm run|chat [options] (legacy mode flags remain supported)");
        out.println();
        out.println("Options:");
        out.println("  --model, -m <path>            required, path to .gguf file");
        out.println("  --interactive, --chat, -i     run in chat mode");
        out.println("  --instruct                    run in instruct (once) mode, default mode");
        out.println("  --prompt, -p <string>         input prompt");
        out.println("  --system-prompt, -sp <string> (optional) system prompt (Llama models)");
        out.println(
                "  --suffix <string>             suffix for fill-in-the-middle request (Codestral)");
        out.println(
                "  --temperature, -temp <float>  temperature in [0,inf], default: auto-detected from model family");
        out.println(
                "  --top-p <float>               p value in top-p (nucleus) sampling in [0,1], default: auto-detected from model family");
        out.println("  --seed <long>                 random seed, default System.nanoTime()");
        out.println(
                "  --ctx-size, -c <int>          context capacity, including prompt and generated tokens");
        out.println(
                "  --max-new-tokens <int>        generated-token limit per request/turn, default: context capacity");
        out.println(
                "  --max-tokens, -n <int>        capacity in positions, prompt plus generated tokens (the context"
                        + " this run allocates; a prompt this long or longer is refused), default "
                        + DEFAULT_MAX_TOKENS);
        out.println(
                "  --stream <boolean>            print tokens during generation; may cause encoding artifacts for non ASCII text, default true");
        out.println(
                "  --echo <boolean>              print ALL tokens to stderr, if true, recommended to set --stream=false, default false");
        out.println(
                "  --with-prefill-decode         enable prefill/decode separation (skip logits during prefill)");
        out.println(
                "  --batch-prefill-size <int>    batched prefill chunk size; requires --with-prefill-decode, must be > 1, enables batched CPU/GPU prefill");
        out.println();
    }

    public static Options getDefaultOptions() {
        String prompt = "Tell me a story with Java"; // Hardcoded for testing
        String systemPrompt = null;
        String suffix = null;
        float temperature = Float.NaN; // resolved from model family after loading
        float topp = Float.NaN; // resolved from model family after loading
        Path modelPath = null;
        long seed = System.nanoTime();
        int maxTokens = DEFAULT_MAX_TOKENS;
        boolean interactive = false;
        boolean stream = true;
        boolean echo = false;
        boolean useTornadoVM = getDefaultTornadoVM();

        return new Options(
                modelPath,
                prompt,
                systemPrompt,
                suffix,
                interactive,
                temperature,
                topp,
                seed,
                maxTokens,
                stream,
                echo,
                useTornadoVM,
                false,
                1);
    }

    public static Options parseOptions(String[] args) {
        String prompt = null;
        String systemPrompt = null;
        String suffix = null;
        float temperature = Float.NaN; // resolved from model family after loading
        float topp = Float.NaN; // resolved from model family after loading
        Path modelPath = null;
        long seed = System.nanoTime();
        int maxTokens = DEFAULT_MAX_TOKENS;
        boolean interactive = false;
        String mode = null;
        int firstArg = 0;
        if (args.length > 0 && (args[0].equals("run") || args[0].equals("chat"))) {
            mode = args[0];
            interactive = mode.equals("chat");
            firstArg = 1;
        }
        Integer maxNewTokens = null;
        boolean contextSpecified = false;
        boolean stream = false;
        boolean echo = false;
        Boolean useTornadovm = null; // null means not specified via command line
        boolean withPrefillDecode = false;
        int batchPrefillSize = 1;

        for (int i = firstArg; i < args.length; i++) {
            String optionName = args[i];
            require(optionName.startsWith("-"), "Invalid option %s", optionName);
            switch (optionName) {
                case "--interactive", "--chat", "-i", "--instruct" -> {
                    String requested = optionName.equals("--instruct") ? "run" : "chat";
                    require(mode == null || mode.equals(requested), "Conflicting run/chat modes");
                    mode = requested;
                    interactive = mode.equals("chat");
                }
                case "--gpu" -> useTornadovm = true;
                case "--with-native-libraries" ->
                        System.setProperty(
                                org.beehive.jllm.runtime.policy.ExecutionPolicy
                                        .NATIVE_LIBRARIES_PROPERTY,
                                "true");
                case "--verbose", "-v" -> System.setProperty("jllm.verbose", "true");
                case "--with-prefill-decode" -> withPrefillDecode = true;
                case "--help", "-h" -> {
                    printUsage(System.out);
                    System.exit(0);
                }
                default -> {
                    String nextArg;
                    if (optionName.contains("=")) {
                        String[] parts = optionName.split("=", 2);
                        optionName = parts[0];
                        nextArg = parts[1];
                    } else {
                        require(i + 1 < args.length, "Missing argument for option %s", optionName);
                        nextArg = args[i + 1];
                        i += 1; // skip arg
                    }
                    switch (optionName) {
                        case "--prompt", "-p" -> prompt = nextArg;
                        case "--system-prompt", "-sp" -> systemPrompt = nextArg;
                        case "--suffix" -> suffix = nextArg;
                        case "--temperature", "--temp" -> temperature = Float.parseFloat(nextArg);
                        case "--top-p" -> topp = Float.parseFloat(nextArg);
                        case "--model", "-m" -> modelPath = Paths.get(nextArg);
                        case "--seed", "-s" -> seed = Long.parseLong(nextArg);
                        case "--ctx-size", "-c", "--max-tokens", "-n" -> {
                            require(
                                    !contextSpecified,
                                    "Specify context capacity once (--ctx-size or --max-tokens)");
                            contextSpecified = true;
                            maxTokens = Integer.parseInt(nextArg);
                            if (optionName.equals("--max-tokens") || optionName.equals("-n")) {
                                System.err.println(
                                        "--max-tokens/-n is deprecated; use --ctx-size/-c (context capacity).");
                            }
                        }
                        case "--max-new-tokens" -> maxNewTokens = Integer.parseInt(nextArg);
                        case "--stream" -> stream = Boolean.parseBoolean(nextArg);
                        case "--echo" -> echo = Boolean.parseBoolean(nextArg);
                        case "--use-tornadovm" -> useTornadovm = Boolean.parseBoolean(nextArg);
                        case "--batch-prefill-size" -> batchPrefillSize = Integer.parseInt(nextArg);
                        default -> require(false, "Unknown option: %s", optionName);
                    }
                }
            }
        }

        require(modelPath != null, "Missing argument: --model <path> is required");
        require(
                !interactive || (prompt == null && suffix == null),
                "chat reads prompts from the terminal; --prompt/--suffix belong to run");

        if (useTornadovm == null) {
            useTornadovm = getDefaultTornadoVM();
        }

        return new Options(
                modelPath,
                prompt,
                systemPrompt,
                suffix,
                interactive,
                temperature,
                topp,
                seed,
                maxTokens,
                stream,
                echo,
                useTornadovm,
                withPrefillDecode,
                batchPrefillSize,
                maxNewTokens == null ? maxTokens : maxNewTokens);
    }
}
