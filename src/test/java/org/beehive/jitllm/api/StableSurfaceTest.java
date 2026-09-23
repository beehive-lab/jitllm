package org.beehive.jitllm.api;

import static org.junit.Assert.assertTrue;

import java.io.IOException;
import java.lang.reflect.Executable;
import java.lang.reflect.GenericArrayType;
import java.lang.reflect.Method;
import java.lang.reflect.Modifier;
import java.lang.reflect.ParameterizedType;
import java.lang.reflect.Type;
import java.lang.reflect.WildcardType;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;
import java.util.TreeMap;
import java.util.TreeSet;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Stream;
import org.junit.Test;

/**
 * <b>A stable member may not expose an experimental type.</b> A caller reading {@code ModelOptions}
 * — no marker — would reasonably assume every method on it is as stable as the class. {@code
 * backend(BackendId)} is not, because {@code BackendId} is still moving. Marking the whole class
 * experimental again would throw away the freeze; marking the <i>member</i> says exactly what is
 * and is not settled.
 *
 * <h2>Why the annotation is read from source</h2>
 *
 * <p>{@code @Experimental} is {@link java.lang.annotation.RetentionPolicy#CLASS} — deliberately, so
 * bytecode tools can see it — which means <b>reflection cannot</b>. Signatures come from
 * reflection, including generic arguments; the markers come from the source. Changing the retention
 * to {@code RUNTIME} to make this test simpler would change a published annotation's contract to
 * suit a test, which is the wrong way round.
 */
public class StableSurfaceTest {

    private static final Path API_SOURCE = Path.of("src/main/java/org/beehive/jitllm/api");
    private static final Path MAIN_SOURCE = Path.of("src/main/java");

    /** The types this release freezes. Changing this set is an API decision, not a refactor. */
    private static final Set<String> STABLE =
            new TreeSet<>(
                    List.of(
                            "LocalModels",
                            "LocalModel",
                            "TextGenerationModel",
                            "GenerationSession",
                            "GenerationRequest",
                            "GenerationResult",
                            "GenerationEvent",
                            "ModelOptions",
                            "SessionOptions",
                            "ChatRole",
                            "ChatContent",
                            "ToolSpec",
                            "ThinkingMode"));

    private static final Set<String> EXPERIMENTAL_API =
            new TreeSet<>(
                    List.of(
                            "ChatMessage",
                            "FinishReason",
                            "GenerationTimings",
                            "ModelConfiguration",
                            "ModelInfo",
                            "InsufficientDeviceMemoryException"));

    /** The exact stable set — a type appearing in neither list is an undeclared decision. */
    @Test
    public void theStableAndExperimentalSetsAreExact() throws IOException {
        Set<String> publicTypes = new TreeSet<>();
        try (Stream<Path> files = Files.list(API_SOURCE)) {
            for (Path p : files.filter(f -> f.toString().endsWith(".java")).toList()) {
                String name = p.getFileName().toString().replace(".java", "");
                String body = Files.readString(p);
                if (Pattern.compile(
                                        "^public (final |abstract |sealed )?(interface|class|enum|record) "
                                                + name,
                                        Pattern.MULTILINE)
                                .matcher(body)
                                .find()
                        || body.contains("public @interface " + name)) {
                    publicTypes.add(name);
                }
            }
        }
        publicTypes.remove("Experimental"); // the marker itself is not part of the surface it marks
        Set<String> declared = new TreeSet<>(STABLE);
        declared.addAll(EXPERIMENTAL_API);
        Set<String> undeclared = new TreeSet<>(publicTypes);
        undeclared.removeAll(declared);
        assertTrue(
                "these public api/** types are in neither the stable nor the experimental set."
                        + " A new public type is an API decision and must be declared in one of them: "
                        + undeclared,
                undeclared.isEmpty());
        Set<String> vanished = new TreeSet<>(declared);
        vanished.removeAll(publicTypes);
        assertTrue(
                "these types are declared but no longer public: " + vanished, vanished.isEmpty());
    }

    /** Every type in the stable set really has no type-level marker, and vice versa. */
    @Test
    public void theMarkersMatchTheDeclaredSets() throws IOException {
        Set<String> annotated = annotatedTypes();
        for (String stable : STABLE) {
            assertTrue(
                    stable + " is declared stable but still carries a type-level @Experimental",
                    !annotated.contains("org.beehive.jitllm.api." + stable));
        }
        for (String experimental : EXPERIMENTAL_API) {
            assertTrue(
                    experimental + " is declared experimental but carries no @Experimental",
                    annotated.contains("org.beehive.jitllm.api." + experimental));
        }
    }

    /**
     * <b>The rule.</b> No stable member exposes an experimental type unless the member says so.
     *
     * <p>Transitive by construction: it walks generic arguments, so a {@code List<ChatMessage>} is
     * caught as surely as a bare {@code ChatMessage}.
     */
    @Test
    public void stableMembersDoNotLeakExperimentalTypes() throws Exception {
        Set<String> experimentalTypes = annotatedTypes();
        Set<String> annotatedMembers = annotatedMemberNames();
        java.util.Map<String, Set<String>> leaks = new TreeMap<>();

        for (String name : STABLE) {
            Class<?> type = Class.forName("org.beehive.jitllm.api." + name);
            List<Class<?>> family = new ArrayList<>(List.of(type));
            family.addAll(List.of(type.getClasses()));
            for (Class<?> c : family) {
                List<Executable> members = new ArrayList<>(List.of(c.getMethods()));
                members.addAll(List.of(c.getConstructors()));
                for (Executable e : members) {
                    if (e.getDeclaringClass().equals(Object.class)
                            || !Modifier.isPublic(e.getModifiers())) {
                        continue;
                    }
                    String memberName =
                            e instanceof Method
                                    ? e.getName()
                                    : e.getDeclaringClass().getSimpleName();
                    if (annotatedMembers.contains(memberName)) {
                        continue; // the member declares its own instability
                    }
                    Set<String> exposed = new TreeSet<>();
                    List<Type> types = new ArrayList<>(List.of(e.getGenericParameterTypes()));
                    if (e instanceof Method m) {
                        types.add(m.getGenericReturnType());
                    }
                    for (Type t : types) {
                        for (Class<?> referenced : flatten(t)) {
                            if (isExperimental(referenced, experimentalTypes)) {
                                exposed.add(referenced.getSimpleName());
                            }
                        }
                    }
                    if (!exposed.isEmpty()) {
                        leaks.put(c.getSimpleName() + "." + memberName, exposed);
                    }
                }
            }
        }
        assertTrue(
                "a stable member must not expose an experimental type without saying so."
                        + " Either mark the member @Experimental, or graduate the type it exposes."
                        + " Leaks: "
                        + leaks,
                leaks.isEmpty());
    }

    /** No backend, TornadoVM, file-format or internal model type appears in a public signature. */
    @Test
    public void noInternalTypeAppearsInThePublicSurface() throws Exception {
        String[] forbidden = {
            "uk.ac.manchester.tornado",
            "org.beehive.jitllm.backend.",
            "org.beehive.jitllm.format.",
            "org.beehive.jitllm.inference.",
            "org.beehive.jitllm.tensor.",
        };
        java.util.Map<String, Set<String>> leaks = new TreeMap<>();
        Set<String> all = new TreeSet<>(STABLE);
        all.addAll(EXPERIMENTAL_API);
        for (String name : all) {
            Class<?> type = Class.forName("org.beehive.jitllm.api." + name);
            List<Class<?>> family = new ArrayList<>(List.of(type));
            family.addAll(List.of(type.getClasses()));
            for (Class<?> c : family) {
                List<Executable> members = new ArrayList<>(List.of(c.getMethods()));
                members.addAll(List.of(c.getConstructors()));
                for (Executable e : members) {
                    if (e.getDeclaringClass().equals(Object.class)
                            || !Modifier.isPublic(e.getModifiers())) {
                        continue;
                    }
                    List<Type> types = new ArrayList<>(List.of(e.getGenericParameterTypes()));
                    if (e instanceof Method m) {
                        types.add(m.getGenericReturnType());
                    }
                    for (Type t : types) {
                        for (Class<?> referenced : flatten(t)) {
                            for (String prefix : forbidden) {
                                if (referenced.getName().startsWith(prefix)) {
                                    leaks.computeIfAbsent(
                                                    c.getSimpleName() + "." + e.getName(),
                                                    k -> new TreeSet<>())
                                            .add(referenced.getName());
                                }
                            }
                        }
                    }
                }
            }
        }
        assertTrue(
                "the public surface must name no backend, TornadoVM, file-format or inference"
                        + " internal. Leaks: "
                        + leaks,
                leaks.isEmpty());
    }

    /**
     * `BackendId` and `ExecutionPolicy` are public API in practice — Quarkus imports both — and are
     * marked accordingly rather than left accidentally stable because they sit outside `api/**`.
     */
    @Test
    public void reachableTypesOutsideTheApiPackageAreMarked() throws IOException {
        Set<String> annotated = annotatedTypes();
        for (String required :
                List.of(
                        "org.beehive.jitllm.runtime.backend.BackendId",
                        "org.beehive.jitllm.runtime.backend.DeviceSelector",
                        "org.beehive.jitllm.runtime.policy.ExecutionPolicy",
                        "org.beehive.jitllm.runtime.policy.StorageOptions",
                        "org.beehive.jitllm.runtime.memory.MemoryPlan",
                        "org.beehive.jitllm.runtime.tensor.DataType",
                        "org.beehive.jitllm.runtime.diagnostics.DiagnosticCode")) {
            assertTrue(
                    required
                            + " is reachable from the public facade and must carry"
                            + " @Experimental; living outside api/** does not make it stable",
                    annotated.contains(required));
        }
    }

    // ── helpers ──────────────────────────────────────────────────────────────

    private static boolean isExperimental(Class<?> c, Set<String> experimentalTypes) {
        String name = c.getName();
        if (experimentalTypes.contains(name)) {
            return true;
        }
        // A nested type inherits its enclosing type's marker: ExecutionPolicy.Overrides is as
        // experimental as ExecutionPolicy, and a member exposing it is exposing that instability.
        int nested = name.indexOf('$');
        return nested > 0 && experimentalTypes.contains(name.substring(0, nested));
    }

    /** Fully-qualified names of types carrying a type-level {@code @Experimental}, from source. */
    private static Set<String> annotatedTypes() throws IOException {
        Set<String> out = new HashSet<>();
        Pattern p =
                Pattern.compile(
                        "@Experimental\\s*\\n\\s*public (?:final |abstract |sealed )?"
                                + "(?:interface|class|enum|record) (\\w+)");
        try (Stream<Path> files = Files.walk(MAIN_SOURCE)) {
            for (Path f : files.filter(x -> x.toString().endsWith(".java")).toList()) {
                String body = Files.readString(f);
                Matcher m = p.matcher(body);
                if (m.find()) {
                    Matcher pkg =
                            Pattern.compile("^package ([\\w.]+);", Pattern.MULTILINE).matcher(body);
                    if (pkg.find()) {
                        out.add(pkg.group(1) + "." + m.group(1));
                    }
                }
            }
        }
        return out;
    }

    /** Simple names of members carrying their own {@code @Experimental}. */
    private static Set<String> annotatedMemberNames() throws IOException {
        Set<String> out = new HashSet<>();
        Pattern p =
                Pattern.compile(
                        "@Experimental\\s*\\n\\s*(?:public )?[\\w<>,.\\[\\] ]+?\\s(\\w+)\\(");
        try (Stream<Path> files = Files.list(API_SOURCE)) {
            for (Path f : files.filter(x -> x.toString().endsWith(".java")).toList()) {
                Matcher m = p.matcher(Files.readString(f));
                while (m.find()) {
                    out.add(m.group(1));
                }
            }
        }
        return out;
    }

    private static Set<Class<?>> flatten(Type t) {
        Set<Class<?>> out = new LinkedHashSet<>();
        if (t instanceof Class<?> c) {
            out.add(c.isArray() ? c.getComponentType() : c);
        } else if (t instanceof ParameterizedType p) {
            out.addAll(flatten(p.getRawType()));
            for (Type arg : p.getActualTypeArguments()) {
                out.addAll(flatten(arg));
            }
        } else if (t instanceof GenericArrayType g) {
            out.addAll(flatten(g.getGenericComponentType()));
        } else if (t instanceof WildcardType w) {
            for (Type b : w.getUpperBounds()) {
                out.addAll(flatten(b));
            }
        }
        return out;
    }
}
