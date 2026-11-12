///usr/bin/env jbang "$0" "$@" ; exit $?
//JAVA 22+
//DEPS io.github.beehive-lab:jitllm:2.0.0-jdk22plus

// TornadoVM itself is not a dependency: it comes from the installed SDK, whose argument file
// carries the module path, the JVMCI/Graal arrangement and the per-backend exports. Those vary
// by TornadoVM version, JDK and backend, so they are taken from the SDK, not listed here.
//JAVA_OPTIONS @${env.TORNADOVM_HOME}/tornado-argfile

// Not supplied by the argfile (mirrors the `jitllm` launcher).
//JAVA_OPTIONS --add-modules jdk.incubator.vector
//JAVA_OPTIONS -Dtornado.tvm.maxbytecodesize=65536
//JAVA_OPTIONS -Duse.tornadovm=true
//JAVA_OPTIONS -Dtornado.enable.fastMathOptimizations=true
//JAVA_OPTIONS -Dtornado.enable.mathOptimizations=false
//JAVA_OPTIONS -Dtornado.enable.nativeFunctions=true
//JAVA_OPTIONS -Dtornado.loop.interchange=true
//JAVA_OPTIONS -Dtornado.device.memory=14GB
//JAVA_OPTIONS -Dtornado.eventpool.maxwaitevents=32000
//JAVA_OPTIONS -Xmx20g

// Same package as JitllmApp, whose main is package-private.
package org.beehive.jitllm;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Properties;

/**
 * JBang entry point for jitllm.
 *
 * <pre>
 *   jbang jitllm@beehive-lab -m model.gguf -p "Tell me a joke"
 *   jbang app install jitllm@beehive-lab &amp;&amp; jitllm -m model.gguf -p "Hello!"
 * </pre>
 *
 * Requires {@code TORNADOVM_HOME} to point at a TornadoVM SDK built for the running JDK.
 */
public class JitllmCli {
    public static void main(String[] args) throws Exception {
        String sdk = System.getenv("TORNADOVM_HOME");
        if (sdk == null || sdk.isBlank()) {
            System.err.println("Error: TORNADOVM_HOME is not set. Install a TornadoVM SDK and export TORNADOVM_HOME.");
            System.exit(1);
        }
        // --fp32-kv-cache is a launcher flag, not a JitllmApp option: turn it into its property.
        List<String> rest = new ArrayList<>();
        for (String arg : args) {
            if (arg.equals("--fp32-kv-cache")) {
                System.setProperty("jitllm.kvcache.fp32", "true");
            } else {
                rest.add(arg);
            }
        }
        boolean metal = backends(sdk).contains("metal-backend");
        // An SDK built with Metal and OpenCL (the usual Apple build) would otherwise run on OpenCL,
        // TornadoVM's default; on Apple silicon Metal is the backend jitllm is tuned for.
        if (metal && System.getProperty("tornado.metal.priority") == null) {
            System.setProperty("tornado.metal.priority", "100");
        }
        // MLX prefill on Metal: dispatch its library tasks asynchronously where the SDK supports it
        // safely (the release after TornadoVM 7.3.1), as the `jitllm` launcher does.
        if (System.getProperty("tornado.mlx.async") == null
                && metal
                && atLeast(tornadoVersion(sdk), 7, 3, 2)) {
            System.setProperty("tornado.mlx.async", "true");
        }
        // The `jitllm` launcher's defaults. An unset --temperature also reaches the sampler as NaN
        // in jitllm 1.0.2, which turns every logit into NaN, so it must always be passed.
        defaultArg(rest, "--temperature", "0.1", "--temp");
        defaultArg(rest, "--top-p", "0.95");
        defaultArg(rest, "--stream", "true");
        JitllmApp.main(rest.toArray(String[]::new));
    }

    private static void defaultArg(List<String> args, String name, String value, String... aliases) {
        for (String arg : args) {
            String option = arg.split("=", 2)[0];
            if (option.equals(name) || List.of(aliases).contains(option)) {
                return;
            }
        }
        args.add(name);
        args.add(value);
    }

    /** The SDK's version, as {@code etc/tornado.release} gives it (e.g. 7.3.1), or empty. */
    private static String tornadoVersion(String sdk) {
        try {
            Properties props = new Properties();
            try (var in = Files.newBufferedReader(Path.of(sdk, "etc", "tornado.release"))) {
                props.load(in);
            }
            return props.getProperty("version", "").trim();
        } catch (IOException e) {
            return "";
        }
    }

    /** Whether a version such as 7.3.2-jdk22plus-dev is at least major.minor.patch. */
    private static boolean atLeast(String version, int major, int minor, int patch) {
        var m = java.util.regex.Pattern.compile("(\\d+)\\.(\\d+)\\.(\\d+)").matcher(version);
        if (!m.lookingAt()) {
            return false;
        }
        int[] have = {Integer.parseInt(m.group(1)), Integer.parseInt(m.group(2)), Integer.parseInt(m.group(3))};
        int[] want = {major, minor, patch};
        for (int i = 0; i < 3; i++) {
            if (have[i] != want[i]) {
                return have[i] > want[i];
            }
        }
        return true;
    }

    /** The backends the SDK was built with, as {@code etc/tornado.backend} lists them. */
    private static String backends(String sdk) {
        try {
            Properties props = new Properties();
            try (var in = Files.newBufferedReader(Path.of(sdk, "etc", "tornado.backend"))) {
                props.load(in);
            }
            return props.getProperty("tornado.backends", "").trim();
        } catch (IOException e) {
            return "";
        }
    }
}
