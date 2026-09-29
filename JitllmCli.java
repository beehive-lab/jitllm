///usr/bin/env jbang "$0" "$@" ; exit $?
//JAVA 22+
//DEPS io.github.beehive-lab:jitllm:1.0.2-jdk22plus

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
        // The Metal kernels have no verified FP16 key/value path, so an FP32 cache is the only one.
        if (System.getProperty("jitllm.kvcache.fp32") == null && backends(sdk).equals("metal-backend")) {
            System.setProperty("jitllm.kvcache.fp32", "true");
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
