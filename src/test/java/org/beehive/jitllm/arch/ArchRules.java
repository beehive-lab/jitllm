package org.beehive.jllm.arch;

import com.tngtech.archunit.core.domain.JavaClass;
import com.tngtech.archunit.core.domain.JavaClasses;
import com.tngtech.archunit.core.domain.JavaModifier;
import java.util.Set;
import java.util.TreeSet;
import java.util.function.Predicate;

/**
 * The dependency rules of {@code docs/architecture/architecture.md}, each expressed as a function
 * from imported classes to the set of violating class names.
 *
 * <p>Returning the violators rather than asserting directly is deliberate: it lets the same rule
 * run against the production tree (compared to an allowlist) and against a deliberately violating
 * fixture (to prove the rule actually bites). A rule that is never seen to fail is not a guardrail.
 */
public final class ArchRules {

    /** TornadoVM's own packages — the thing Rules 1 and 2 keep out of the upper layers. */
    public static final String TORNADO_VM = "uk.ac.manchester.tornado";

    /** The Tornado backend package. */
    public static final String TORNADO_BACKEND = "org.beehive.jllm.backend.tornado";

    public static final String MODEL = "org.beehive.jllm.model";

    /** The metrics seam — Rule 17's runtime-layer interface package. */
    public static final String RUNTIME_METRICS = "org.beehive.jllm.runtime.metrics";

    /** The runtime's tensor vocabulary — Rule 4's format-free side. */
    public static final String RUNTIME_TENSOR = "org.beehive.jllm.runtime.tensor";

    /** GGUF and GGML types: file representation, which the runtime layer must not name. */
    public static final String FORMAT_TYPES = "org.beehive.jllm.format";

    /** The backend-neutral program layer — Rule 3's subject. Today: the operation vocabulary. */
    public static final String PROGRAM = "org.beehive.jllm.program";

    /** The runtime layer, in scope for Rule 14 alongside the program layer. */
    public static final String RUNTIME = "org.beehive.jllm.runtime";

    public static final Set<String> GENERATION_PACKAGES =
            Set.of(
                    "org.beehive.jllm.tokenizer",
                    "org.beehive.jllm.model.format",
                    "org.beehive.jllm.inference.sampler",
                    "org.beehive.jllm.generation");

    /** Rule 11's type-specific list — the plan types most likely to leak out of the backend. */
    public static final Set<String> PLAN_TYPES =
            Set.of(
                    "uk.ac.manchester.tornado.api.TaskGraph",
                    "uk.ac.manchester.tornado.api.ImmutableTaskGraph",
                    "uk.ac.manchester.tornado.api.TornadoExecutionPlan",
                    "uk.ac.manchester.tornado.api.GridScheduler");

    private ArchRules() {}

    /** Rule 1 — TornadoVM stays in the Tornado backend. */
    public static Set<String> rule1TornadoVmOutsideBackend(
            JavaClasses classes, String backendPrefix) {
        return violators(
                classes, c -> !inPackage(c, backendPrefix) && dependsOnPackage(c, TORNADO_VM));
    }

    /**
     * Rule 2 — model architecture packages do not import TornadoVM. Broader than Rule 1 inside
     * {@code model}: depending on the backend package counts too, because {@code
     * TornadoVMMasterPlan} is Tornado-specific without being a {@code uk.ac.manchester} import.
     */
    public static Set<String> rule2ModelDependsOnTornado(
            JavaClasses classes, String modelPrefix, String backendPrefix) {
        return violators(
                classes,
                c ->
                        inPackage(c, modelPrefix)
                                && (dependsOnPackage(c, TORNADO_VM)
                                        || dependsOnPackage(c, backendPrefix)));
    }

    public static final String CPU_BACKEND = "org.beehive.jllm.backend.cpu";

    /**
     * It would have been possible to satisfy this by moving {@code InferenceCore} into {@code
     * backend.cpu} and leaving the families pointing at it. This is the assertion that would have
     * caught that, and it has <b>no allowlist</b>: an entry here would mean a family had gone back
     * to naming what executes it.
     */
    public static Set<String> m12dfModelNamesTheCpuBackend(
            JavaClasses classes, String modelPrefix) {
        return violators(
                classes, c -> inPackage(c, modelPrefix) && dependsOnPackage(c, CPU_BACKEND));
    }

    /**
     * Rule 5 — models own immutable configuration and weights. Scoped to loaded-model types;
     * loaders and builders are explicitly out of scope per the rule text.
     */
    public static Set<String> rule5MutableModelFields(
            JavaClasses classes, Predicate<JavaClass> loadedModelTypes) {
        return violators(classes, c -> loadedModelTypes.test(c) && hasNonFinalField(c));
    }

    /** The engine tier's package. Literal, because {@code inPackage} is a prefix match. */
    public static final String ENGINE = "org.beehive.jllm.engine";

    /** Rule 18 — nothing below the engine may depend on it. */
    public static Set<String> rule18LowerTiersReachEngine(
            JavaClasses classes, String... lowerPrefixes) {
        return violators(
                classes,
                c -> {
                    for (String p : lowerPrefixes) {
                        if (inPackage(c, p)) {
                            return c.getDirectDependenciesFromSelf().stream()
                                    .anyMatch(d -> inPackage(d.getTargetClass(), ENGINE));
                        }
                    }
                    return false;
                });
    }

    /** The backend-neutral SPI package. Everything above the backend may name this. */
    public static final String RUNTIME_BACKEND = "org.beehive.jllm.runtime.backend";

    /**
     * {@code runtime.backend} holds identities, selectors, options and contracts; the
     * implementations live in {@code backend.cpu} and {@code backend.tornado}. If the neutral
     * package could name an implementation, the arrangement would be decorative: the upper layers
     * would reach a backend transitively through the very type that exists to keep them from doing
     * so.
     *
     * <p>TornadoVM itself is covered by Rule 1, which already forbids it outside the Tornado
     * backend. This rule adds the project's own implementation packages, which Rule 1 does not
     * mention because they are not {@code uk.ac.manchester} imports.
     *
     * @param neutralPrefix the neutral SPI package — {@link #RUNTIME_BACKEND} in production, a
     *     fixture package in the self-test
     * @param implementationPrefixes the implementation packages — today {@code tornadovm} (the
     *     stand-in for {@code backend.tornado}), and {@code backend}
     */
    public static Set<String> m12dbNeutralSpiDependsOnAnImplementation(
            JavaClasses classes, String neutralPrefix, String... implementationPrefixes) {
        return violators(
                classes,
                c -> {
                    if (!inPackage(c, neutralPrefix)) {
                        return false;
                    }
                    if (dependsOnPackage(c, TORNADO_VM)) {
                        return true;
                    }
                    for (String p : implementationPrefixes) {
                        if (dependsOnPackage(c, p)) {
                            return true;
                        }
                    }
                    return false;
                });
    }

    /**
     * Rule 1 would not catch this on its own: {@code TornadoKvStore} is a project type, not a
     * {@code uk.ac.manchester} import, so a class could name it and keep Rule 1 green. This is the
     * assertion that {@code State}'s old {@code instanceof TornadoKvStore} cannot come back
     * anywhere, under any name.
     *
     * @param backendPrefix the backend package the cast is allowed to live in
     * @param storageTypes the backend storage types, by simple name
     */
    public static Set<String> m12dgStorageResolvedOutsideTheBackend(
            JavaClasses classes, String backendPrefix, Set<String> storageTypes) {
        return violators(
                classes,
                c ->
                        !inPackage(c, backendPrefix)
                                && c.getDirectDependenciesFromSelf().stream()
                                        .anyMatch(
                                                d ->
                                                        storageTypes.contains(
                                                                d.getTargetClass()
                                                                        .getSimpleName())));
    }

    /** Rule 7 — KV storage is never reachable from a model or a program. */
    public static Set<String> rule7ModelReachesKvStorage(
            JavaClasses classes, String... upperPrefixes) {
        return violators(
                classes,
                c -> {
                    for (String p : upperPrefixes) {
                        if (inPackage(c, p)) {
                            return c.getDirectDependenciesFromSelf().stream()
                                    .anyMatch(
                                            d -> {
                                                String n = d.getTargetClass().getSimpleName();
                                                return n.contains("KvCache")
                                                        || n.contains("BlockPool");
                                            });
                        }
                    }
                    return false;
                });
    }

    /**
     * Today's stand-in for {@code.api.} and {@code.integration.}: the CLI entry point, the CLI
     * options record and the HTTP server. There is no {@code integration.cli} package yet, so the
     * CLI is identified by type rather than package.
     */
    public static final Set<String> CLI_TYPES =
            Set.of("org.beehive.jllm.JllmApp", "org.beehive.jllm.Options");

    public static final String CLI = "org.beehive.jllm.integration.cli";

    public static final String SERVER = "org.beehive.jllm.server";

    /**
     * The runnable API examples. They are an integration like the CLI: their whole purpose is to
     * show a caller what the facade looks like in use, and printing the answer is the showing.
     */
    public static final String EXAMPLES = "org.beehive.jllm.examples";

    /**
     * Rule 8a defines generation policy as "the token loop, stop conditions, streaming, transport,
     * console I/O" and allows {@code generation.**} → {@code model.**}. So this package is not a
     * violator of either rule: it is the layer both rules describe, and recognizing it is what the
     * rules always meant. Before it existed the loops lived on {@code Model}, which is why {@code
     * Model} was in both allowlists — and why Rule 16's entry for it is now stale.
     */
    public static final String GENERATION = "org.beehive.jllm.generation";

    /**
     * Rule 8a — generation policy is separate from forward execution. Lower layers must not reach
     * the CLI, the options record or the server integration. The integrations themselves are
     * excluded: depending on generation policy is their job.
     */
    public static Set<String> rule8aLowerLayersDependOnGenerationPolicy(JavaClasses classes) {
        return violators(
                classes,
                c ->
                        !isIntegration(c)
                                && c.getDirectDependenciesFromSelf().stream()
                                        .anyMatch(
                                                d -> {
                                                    String n = d.getTargetClass().getName();
                                                    return CLI_TYPES.contains(n)
                                                            || n.startsWith(SERVER + ".")
                                                            || n.startsWith(CLI + ".");
                                                }));
    }

    /**
     * Rule 15 — no central model-type switches for new architectures.
     *
     * <p>Adding an architecture should mean adding a provider, not editing switch statements spread
     * across packages. Dispatch on {@code ModelType} is what this targets, not the enum's
     * existence: it is expected to survive as an internal identifier long after loading moves to
     * the provider SPI, and the legacy path still selects it with {@code -Djllm.providers=false}.
     *
     * <p>The provider package is exempt because that is where enumerating families is the point.
     */
    public static Set<String> rule15ModelTypeDispatchOutsideProviders(JavaClasses classes) {
        return violators(
                classes,
                c ->
                        !inPackage(c, PROVIDERS)
                                && !c.getName().equals(MODEL_TYPE)
                                && !c.getName().startsWith(MODEL_TYPE + "$")
                                && modelTypeConstantsRead(c) > 1);
    }

    /**
     * How many distinct {@code ModelType} constants a class reads.
     *
     * <p>This is what separates dispatch from identity. A family class naming its own constant in
     * {@code getModelType()} reads exactly one and is not the target: the enum surviving as an
     * internal identifier is expected. A class that reads several is choosing behaviour by family,
     * which is the switch Rule 15 exists to remove — including the synthetic {@code $SwitchMap}
     * class javac generates for a switch over the enum, which reads every constant.
     */
    private static int modelTypeConstantsRead(JavaClass c) {
        return (int)
                c.getFieldAccessesFromSelf().stream()
                        .filter(a -> a.getTargetOwner().getName().equals(MODEL_TYPE))
                        .map(a -> a.getTarget().getName())
                        .distinct()
                        .count();
    }

    /**
     * Rule 16 — no console I/O outside the CLI integration. Library code that prints cannot be
     * silenced or routed by an embedder.
     */
    public static Set<String> rule16ConsoleIoOutsideCli(JavaClasses classes) {
        return violators(classes, c -> !isIntegration(c) && printsToConsole(c));
    }

    private static boolean isIntegration(JavaClass c) {
        String outer = c.getName().split("\\$")[0];
        return CLI_TYPES.contains(outer)
                || inPackage(c, CLI)
                || inPackage(c, SERVER)
                || inPackage(c, EXAMPLES)
                || inPackage(c, GENERATION);
    }

    private static boolean printsToConsole(JavaClass c) {
        return c.getMethodCallsFromSelf().stream()
                .anyMatch(
                        call -> {
                            String m = call.getName();
                            return call.getTargetOwner().getName().equals("java.io.PrintStream")
                                    && (m.equals("println")
                                            || m.equals("print")
                                            || m.equals("printf"));
                        });
    }

    /**
     * Rule 11 — TaskGraph / ImmutableTaskGraph / TornadoExecutionPlan / GridScheduler stay in the
     * backend.
     */
    public static Set<String> rule11PlanTypesOutsideBackend(
            JavaClasses classes, String backendPrefix) {
        return violators(
                classes,
                c ->
                        !inPackage(c, backendPrefix)
                                && c.getDirectDependenciesFromSelf().stream()
                                        .anyMatch(
                                                d ->
                                                        PLAN_TYPES.contains(
                                                                d.getTargetClass().getName())));
    }

    /**
     * Rule 17 — the metrics seam depends on nothing in this project.
     *
     * <p>It is written from below (backends) and read from above (engine, API), so it can only stay
     * callable from both directions by depending on neither. A dependency out of this package — on
     * a backend, a model, or a sink implementation — turns the one designed upward edge into an
     * ordinary cycle. Dependencies on the JDK and on the package's own types are what remains.
     */
    public static Set<String> rule17MetricsSeamDependsOnNothing(
            JavaClasses classes, String metricsPrefix) {
        return violators(
                classes,
                c ->
                        inPackage(c, metricsPrefix)
                                && c.getDirectDependenciesFromSelf().stream()
                                        .anyMatch(
                                                d -> {
                                                    JavaClass target =
                                                            d.getTargetClass()
                                                                    .getBaseComponentType();
                                                    String name = target.getName();
                                                    return !inPackage(target, metricsPrefix)
                                                            && !name.startsWith("java.")
                                                            && !target.isPrimitive();
                                                }));
    }

    /**
     * The format layer's own home today: GGUF, GGMLType and the tensor classes that decode them.
     */
    public static final String FORMAT_LAYER = "org.beehive.jllm.format";

    /** Loading is where format and runtime meet, so the loading path is in scope by design. */
    public static final String LOADERS = "org.beehive.jllm.model.loader";

    /**
     * The provider SPI is the loading path too — recognizing a file and reading it is what a
     * provider is for, so naming the format there is the rule working, not failing.
     */
    public static final String PROVIDERS = "org.beehive.jllm.model.provider";

    /** Dispatching on this is Rule 15's subject; the enum itself is allowed to exist. */
    public static final String MODEL_TYPE = "org.beehive.jllm.model.ModelType";

    /** Rule 4 — GGUF's types stay in the format layer and the loading path. */
    public static Set<String> rule4FormatTypesOutsideFormatAndLoaders(JavaClasses classes) {
        return violators(
                classes,
                c ->
                        !inPackage(c, FORMAT_LAYER)
                                && !inPackage(c, LOADERS)
                                && !inPackage(c, PROVIDERS)
                                && !c.getModifiers().contains(JavaModifier.SYNTHETIC)
                                && (namesFormatType(c) || callsMethodReturningFormatType(c)));
    }

    private static boolean namesFormatType(JavaClass c) {
        return c.getDirectDependenciesFromSelf().stream()
                .anyMatch(d -> isFormatType(d.getTargetClass().getBaseComponentType().getName()));
    }

    /**
     * A local variable's type is not a dependency ArchUnit records, so a class that only writes
     * {@code GGMLType t = weights.getWeightType();} would slip past. Calling a method that
     * <i>returns</i> a format type is the same leak and is counted as one.
     */
    private static boolean callsMethodReturningFormatType(JavaClass c) {
        return c.getMethodCallsFromSelf().stream()
                .anyMatch(
                        call ->
                                call.getTarget().getRawReturnType() != null
                                        && isFormatType(
                                                call.getTarget()
                                                        .getRawReturnType()
                                                        .getBaseComponentType()
                                                        .getName()));
    }

    private static boolean isFormatType(String name) {
        return name.equals(FORMAT_LAYER + ".GGMLType")
                || name.equals(FORMAT_LAYER + ".GGUF")
                || name.equals(FORMAT_LAYER + ".GGMLTensorEntry");
    }

    /**
     * Rule 4 — the runtime's tensor vocabulary names no file-format type and no backend type.
     *
     * <p>Scoped to the new runtime packages rather than to the whole tree: Rule 4's full form still
     * has an allowlist to shrink (emptied at). What this catches is the drift that would make that
     * work pointless — a format type reaching the very packages introduced to be free of it.
     */
    public static Set<String> rule4RuntimeNamesFormatOrBackendTypes(
            JavaClasses classes, String runtimePrefix) {
        return violators(
                classes,
                c ->
                        inPackage(c, runtimePrefix)
                                && (dependsOnPackage(c, FORMAT_TYPES)
                                        || dependsOnPackage(c, TORNADO_VM)
                                        || dependsOnPackage(c, TORNADO_BACKEND)));
    }

    /**
     * Rule 3 — the backend-neutral program layer does not import TornadoVM or a backend.
     *
     * <p>A program is a description. If it references {@code TaskGraph}, {@code GridScheduler} or
     * {@code ImmutableTaskGraph}, it is not a description — it is TornadoVM code, and no second
     * backend can ever implement it.
     */
    public static Set<String> rule3ProgramImportsBackend(
            JavaClasses classes, String programPrefix, String backendPrefix) {
        return violators(
                classes,
                c ->
                        inPackage(c, programPrefix)
                                && (dependsOnPackage(c, TORNADO_VM)
                                        || dependsOnPackage(c, backendPrefix)));
    }

    /**
     * Rule 14 — core abstractions do not assume generation.
     *
     * <p>The program, runtime and backend layers must not <i>require</i> a tokenizer, a chat
     * format, a sampler or a generation loop. Those are capabilities a text-generation model adds;
     * a vocabulary that assumed them could not serve embeddings, classification or reranking
     * without a second framework.
     *
     * <p><b>Not in tension with Rule 8b.</b> Sampling is an operation and may execute on the
     * device, so an operation type <i>named</i> {@code Sample} is fine. What this rule catches is a
     * core type depending on the generation machinery — a sampler implementation, a chat format, a
     * tokenizer — which is a different thing from naming the work.
     */
    public static Set<String> rule14CoreAssumesGeneration(
            JavaClasses classes, String... corePrefixes) {
        return violators(
                classes,
                c -> {
                    boolean core = false;
                    for (String prefix : corePrefixes) {
                        core |= inPackage(c, prefix);
                    }
                    if (!core) {
                        return false;
                    }
                    return c.getDirectDependenciesFromSelf().stream()
                            .anyMatch(
                                    d -> {
                                        JavaClass target =
                                                d.getTargetClass().getBaseComponentType();
                                        String pkg = target.getPackageName();
                                        for (String generation : GENERATION_PACKAGES) {
                                            if (pkg.equals(generation)
                                                    || pkg.startsWith(generation + ".")) {
                                                return true;
                                            }
                                        }
                                        return target.getSimpleName().contains("ChatFormat");
                                    });
                });
    }

    // helpers

    private static Set<String> violators(JavaClasses classes, Predicate<JavaClass> violates) {
        Set<String> out = new TreeSet<>();
        for (JavaClass c : classes) {
            if (violates.test(c)) {
                out.add(c.getName());
            }
        }
        return out;
    }

    private static boolean inPackage(JavaClass c, String prefix) {
        String p = c.getPackageName();
        return p.equals(prefix) || p.startsWith(prefix + ".");
    }

    private static boolean dependsOnPackage(JavaClass c, String prefix) {
        return c.getDirectDependenciesFromSelf().stream()
                .anyMatch(
                        d -> {
                            String p = d.getTargetClass().getPackageName();
                            return p.equals(prefix) || p.startsWith(prefix + ".");
                        });
    }

    private static boolean hasNonFinalField(JavaClass c) {
        return c.getFields().stream().anyMatch(f -> !f.getModifiers().contains(JavaModifier.FINAL));
    }
}
