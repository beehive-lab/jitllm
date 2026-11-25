///usr/bin/env jbang "$0" "$@" ; exit $?
//JAVA 22+
//DEPS io.github.beehive-lab:jitllm:3.1.0-jdk22plus
//DEPS io.github.beehive-lab:tornado-runtime:8.2.0-jdk22plus
//DEPS io.github.beehive-lab:tornado-drivers-cuda:8.2.0-jdk22plus
//DEPS io.github.beehive-lab:tornado-drivers-opencl:8.2.0-jdk22plus
//DEPS io.github.beehive-lab:tornado-drivers-metal:8.2.0-jdk22plus
//DEPS io.github.beehive-lab:tornado-cublas:8.2.0-jdk22plus
//DEPS io.github.beehive-lab:tornado-cudnn:8.2.0-jdk22plus
//DEPS io.github.beehive-lab:tornado-mlx:8.2.0-jdk22plus
/*
 * Copyright (c) 2024-2026, APT Group, Department of Computer Science,
 * The University of Manchester.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

// TornadoVM comes from Maven Central like jitllm: its runtime and the CUDA, OpenCL and Metal
// drivers reach each vendor's driver through the foreign-function API, so no TornadoVM SDK is
// needed. A driver whose vendor library is absent finds no devices. The cuBLAS, cuDNN and MLX
// bindings are used where those libraries are installed; otherwise jitllm keeps its own kernels.
//JAVA_OPTIONS --enable-native-access=ALL-UNNAMED

// The `jitllm` launcher's JVM settings.
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

import java.util.ArrayList;
import java.util.List;

/**
 * JBang entry point for jitllm.
 *
 * <pre>
 *   jbang jitllm@beehive-lab -m model.gguf -p "Tell me a joke"
 *   jbang app install jitllm@beehive-lab &amp;&amp; jitllm -m model.gguf -p "Hello!"
 *   jbang jitllm@beehive-lab --opencl -m model.gguf -p "Hello!"     # another backend
 * </pre>
 *
 * Needs JDK 22 or newer and a GPU driver (CUDA, OpenCL or Metal); no TornadoVM SDK.
 */
public class JitllmCli {
    public static void main(String[] args) throws Exception {
        // Launcher flags that are not JitllmApp options: turn them into their properties.
        List<String> rest = new ArrayList<>();
        String backend = isAppleSilicon() ? "metal" : "cuda";
        for (String arg : args) {
            switch (arg) {
                case "--fp32-kv-cache" -> System.setProperty("jitllm.kvcache.fp32", "true");
                case "--cuda", "--opencl", "--metal" -> backend = arg.substring(2);
                default -> rest.add(arg);
            }
        }
        // All three drivers are on the class path, and TornadoVM ranks OpenCL first by default.
        // Rank the chosen backend first, then CUDA, OpenCL, Metal, as the `jitllm` launcher does; a
        // backend with no device on this machine is skipped.
        List<String> order = new ArrayList<>(List.of("cuda", "opencl", "metal"));
        order.remove(backend);
        order.addFirst(backend);
        for (int i = 0; i < order.size(); i++) {
            String property = "tornado." + order.get(i) + ".priority";
            if (System.getProperty(property) == null) {
                System.setProperty(property, String.valueOf(order.size() - i));
            }
        }
        if (System.getProperty("tornado.opencl.compiler.flags") == null) {
            System.setProperty("tornado.opencl.compiler.flags",
                    "-cl-denorms-are-zero -cl-no-signed-zeros -cl-finite-math-only");
        }
        boolean metal = backend.equals("metal");
        // MLX prefill on Metal: dispatch its library tasks asynchronously, as the `jitllm` launcher
        // does (safe from the TornadoVM release after 7.3.1; the version above is 8.1.0).
        if (System.getProperty("tornado.mlx.async") == null && metal) {
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

    /** macOS on an arm64 processor, where Metal is the GPU backend. */
    private static boolean isAppleSilicon() {
        String os = System.getProperty("os.name", "").toLowerCase();
        String arch = System.getProperty("os.arch", "");
        return os.contains("mac") && (arch.equals("aarch64") || arch.equals("arm64"));
    }
}
