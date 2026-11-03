package org.beehive.jitllm.integration.nativeimage;

import java.io.IOException;
import java.io.InputStream;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.jar.JarEntry;
import java.util.jar.JarFile;
import java.util.stream.Stream;
import org.graalvm.nativeimage.hosted.Feature;
import org.graalvm.nativeimage.hosted.RuntimeReflection;
import org.graalvm.nativeimage.hosted.RuntimeSerialization;

/**
 * Registers what TornadoVM needs from jitllm's own classes in a GraalVM native image. Enabled by
 * {@code META-INF/native-image/io.github.beehive-lab/jitllm/native-image.properties}, so any image
 * built with the jitllm jar on its class path (the CLI, a Quarkus native application) picks it up;
 * on a JVM this class is never loaded.
 *
 * <p>TornadoVM turns a {@code TaskGraph.task(...)} lambda into a kernel through the lambda's {@code
 * writeReplace}: the {@code SerializedLambda} names the kernel's class, which TornadoVM loads by
 * name, and its compiler then reads that class's bytecode. In an image this needs three things the
 * closed-world analysis cannot see:
 *
 * <ul>
 *   <li>every class that creates a serializable lambda registered as lambda-capturing, so the
 *       lambda keeps {@code writeReplace} — exactly the classes javac gave a {@code
 *       $deserializeLambda$} method;
 *   <li>the kernel classes loadable by name — every jitllm class is registered, since a kernel can
 *       call helpers in any package — and their declared methods, which TornadoVM after 8.0.0 looks
 *       up by reflection; classes that use the Vector API (disabled in the image, see the
 *       properties file) hold no kernels and keep only the class registration;
 *   <li>their class files as resources — the {@code org/beehive/jitllm/**}{@code /*.class} glob in
 *       the {@code reachability-metadata.json} next to the properties file.
 * </ul>
 *
 * <p>Scanning the class files here, instead of listing the classes in JSON, keeps a model family
 * that was never traced working: a list recorded by the tracing agent only holds the kernels the
 * traced runs compiled.
 */
public final class JitllmNativeImageFeature implements Feature {

    private static final String PACKAGE_PATH = "org/beehive/jitllm/";
    private static final byte[] DESERIALIZE_LAMBDA =
            "$deserializeLambda$".getBytes(StandardCharsets.UTF_8);
    private static final byte[] VECTOR_API =
            "jdk/incubator/vector/".getBytes(StandardCharsets.UTF_8);

    @Override
    public String getDescription() {
        return "Registers jitllm's TornadoVM kernels and task lambdas for reflection and"
                + " serialization";
    }

    @Override
    public void beforeAnalysis(BeforeAnalysisAccess access) {
        int classes = 0;
        for (Path entry : access.getApplicationClassPath()) {
            for (ClassFile file : classFiles(entry)) {
                Class<?> type = access.findClassByName(file.name());
                if (type == null) {
                    continue;
                }
                RuntimeReflection.register(type);
                classes++;
                // This class holds both markers as string constants, not as references.
                if (type == JitllmNativeImageFeature.class) {
                    continue;
                }
                if (indexOf(file.bytes(), VECTOR_API) < 0) {
                    RuntimeReflection.register(type.getDeclaredMethods());
                }
                if (indexOf(file.bytes(), DESERIALIZE_LAMBDA) >= 0) {
                    RuntimeSerialization.registerLambdaCapturingClass(type);
                }
            }
        }
        if (classes == 0) {
            throw new IllegalStateException(
                    "jitllm native-image feature found no "
                            + PACKAGE_PATH
                            + " classes on the application class path");
        }
    }

    private record ClassFile(String name, byte[] bytes) {}

    private static List<ClassFile> classFiles(Path entry) {
        List<ClassFile> files = new ArrayList<>();
        try {
            if (Files.isDirectory(entry)) {
                Path root = entry.resolve(PACKAGE_PATH);
                if (!Files.isDirectory(root)) {
                    return files;
                }
                try (Stream<Path> walk = Files.walk(root)) {
                    for (Path p : (Iterable<Path>) walk::iterator) {
                        String relative = entry.relativize(p).toString().replace('\\', '/');
                        if (isClassFile(relative)) {
                            files.add(new ClassFile(className(relative), Files.readAllBytes(p)));
                        }
                    }
                }
            } else if (entry.toString().endsWith(".jar") && Files.isRegularFile(entry)) {
                try (JarFile jar = new JarFile(entry.toFile())) {
                    for (JarEntry e : (Iterable<JarEntry>) jar.stream()::iterator) {
                        if (isClassFile(e.getName())) {
                            try (InputStream in = jar.getInputStream(e)) {
                                files.add(new ClassFile(className(e.getName()), in.readAllBytes()));
                            }
                        }
                    }
                }
            }
        } catch (IOException e) {
            throw new UncheckedIOException("cannot scan " + entry + " for jitllm classes", e);
        }
        return files;
    }

    private static boolean isClassFile(String path) {
        return path.startsWith(PACKAGE_PATH)
                && path.endsWith(".class")
                && !path.endsWith("module-info.class");
    }

    private static String className(String path) {
        return path.substring(0, path.length() - ".class".length()).replace('/', '.');
    }

    private static int indexOf(byte[] haystack, byte[] needle) {
        outer:
        for (int i = 0; i <= haystack.length - needle.length; i++) {
            for (int j = 0; j < needle.length; j++) {
                if (haystack[i + j] != needle[j]) {
                    continue outer;
                }
            }
            return i;
        }
        return -1;
    }
}
