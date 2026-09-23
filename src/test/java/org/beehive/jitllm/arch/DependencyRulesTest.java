package org.beehive.jitllm.arch;

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import com.tngtech.archunit.core.domain.JavaClass;
import com.tngtech.archunit.core.domain.JavaClasses;
import java.util.Set;
import java.util.TreeSet;
import org.junit.Test;

/**
 * Each rule is compared to its allowlist in <b>both</b> directions. A class that violates without
 * being listed fails the build, which is the point of the rule; a listed class that no longer
 * violates <i>also</i> fails, because stale entries hide progress and would let an allowlist
 * quietly stop shrinking (policy item 5).
 */
public class DependencyRulesTest {

    private static JavaClasses classes() {
        return ProductionClasses.get();
    }

    @Test
    public void rule1_tornadoVmStaysInTheTornadoBackend() {
        assertMatchesAllowlist(
                "Rule 1 (TornadoVM outside the Tornado backend)",
                ArchRules.rule1TornadoVmOutsideBackend(classes(), ArchRules.TORNADO_BACKEND),
                Allowlists.RULE_1);
    }

    @Test
    public void rule2_modelPackagesDoNotImportTornado() {
        assertMatchesAllowlist(
                "Rule 2 (model depends on TornadoVM or the Tornado backend)",
                ArchRules.rule2ModelDependsOnTornado(
                        classes(), ArchRules.MODEL, ArchRules.TORNADO_BACKEND),
                Allowlists.RULE_2);
    }

    /**
     * <b>No allowlist.</b> It passes on today's code, and an entry here would mean a family had
     * gone back to naming what executes it.
     */
    @Test
    public void m12df_noModelClassNamesTheCpuBackend() {
        Set<String> violations = ArchRules.m12dfModelNamesTheCpuBackend(classes(), ArchRules.MODEL);
        assertTrue(
                "a model architecture describes a transformer; naming the thing that"
                        + " executes it on the host is someone else's job. Violations: "
                        + violations,
                violations.isEmpty());
    }

    /**
     * Rule 17 says the seam depends on nothing. It does not say that <b>nothing depends on an
     * exporter</b>, and that is the direction a regression would actually come from: a renderer
     * named in the decode loop, or a {@code printMetrics()} call added where the numbers happen to
     * be convenient.
     *
     * <p>The consumers live in {@code auxiliary.metrics} and are reached through {@link
     * org.beehive.jitllm.runtime.metrics.MetricsSink}. Inference, the backends, the runtime and the
     * program layer record <b>through the seam</b>; formatting and I/O belong to the layer that
     * owns output.
     */
    @Test
    public void t133_coreNamesNoMetricsExporter() {
        Set<String> violations = new TreeSet<>();
        for (JavaClass c : classes()) {
            String name = c.getName();
            boolean isCore =
                    name.startsWith("org.beehive.jitllm.inference.")
                            || name.startsWith("org.beehive.jitllm.backend.")
                            || name.startsWith("org.beehive.jitllm.runtime.")
                            || name.startsWith("org.beehive.jitllm.program.");
            if (!isCore) {
                continue;
            }
            for (var dependency : c.getDirectDependenciesFromSelf()) {
                String target = dependency.getTargetClass().getBaseComponentType().getName();
                if (target.startsWith("org.beehive.jitllm.auxiliary.metrics.")) {
                    violations.add(name + " -> " + target);
                }
            }
        }
        assertTrue(
                "core records through MetricsSink; naming an exporter from inference, a backend,"
                        + " the runtime or the program layer inverts the metrics seam."
                        + " Violations: "
                        + violations,
                violations.isEmpty());
    }

    @Test
    public void rule5_loadedModelsHaveOnlyFinalFields() {
        assertMatchesAllowlist(
                "Rule 5 (mutable fields on loaded-model types)",
                ArchRules.rule5MutableModelFields(
                        classes(), DependencyRulesTest::isLoadedModelType),
                Allowlists.RULE_5);
    }

    @Test
    public void rule7_kvStorageIsNotReachableFromModels() {
        Set<String> violations = ArchRules.rule7ModelReachesKvStorage(classes(), ArchRules.MODEL);
        assertTrue(
                "Rule 7 passes today and must stay passing; new violations: " + violations,
                violations.isEmpty());
    }

    /**
     * Rule 18 — the engine sits above sessions and below the public API, and nothing below it may
     * reach it.
     */
    @Test
    public void rule18_lowerTiersDoNotReachTheEngine() {
        Set<String> violations =
                ArchRules.rule18LowerTiersReachEngine(
                        classes(),
                        ArchRules.MODEL,
                        "org.beehive.jitllm.runtime",
                        "org.beehive.jitllm.inference",
                        "org.beehive.jitllm.backend.tornado",
                        "org.beehive.jitllm.tensor");
        assertTrue(
                "Rule 18: a model, session, runtime or backend type reached into ..engine..;"
                        + " the simple path must stay usable without a scheduler. Violations: "
                        + violations,
                violations.isEmpty());
    }

    @Test
    public void m12db_theNeutralBackendSpiNamesNoImplementation() {
        Set<String> violations =
                ArchRules.m12dbNeutralSpiDependsOnAnImplementation(
                        classes(),
                        ArchRules.RUNTIME_BACKEND,
                        ArchRules.TORNADO_BACKEND,
                        "org.beehive.jitllm.backend");
        assertTrue(
                "runtime.backend must hold neutral contracts only, and must not name"
                        + " TornadoVM or an implementation package. Violations: "
                        + violations,
                violations.isEmpty());
    }

    @Test
    public void m12dg_backendStorageIsResolvedOnlyInsideTheBackend() {
        Set<String> violations =
                ArchRules.m12dgStorageResolvedOutsideTheBackend(
                        classes(), ArchRules.TORNADO_BACKEND, Set.of("TornadoKvStore"));
        assertTrue(
                "only the backend may resolve a lease or store to backend storage."
                        + " Violations: "
                        + violations,
                violations.isEmpty());
    }

    @Test
    public void rule11_planTypesStayInTheTornadoBackend() {
        Set<String> violations =
                ArchRules.rule11PlanTypesOutsideBackend(classes(), ArchRules.TORNADO_BACKEND);
        assertTrue(
                "Rule 11 passes today and must stay passing; new violations: " + violations,
                violations.isEmpty());
    }

    /**
     * The rule that used to guard it asked whether a temporary type had leaked into public surface.
     * The type is gone, so the question is answered permanently and the assertion is simply that it
     * stays gone — a stopgap that no longer exists cannot become permanent.
     */
    @Test
    public void theLoadTargetAdapterIsGone() {
        Set<String> survivors =
                classes().stream()
                        .map(c -> c.getName())
                        .filter(name -> name.endsWith(".LoadTarget"))
                        .collect(java.util.stream.Collectors.toSet());
        assertTrue("LoadTarget was deleted; found: " + survivors, survivors.isEmpty());
    }

    /** Rule 4 — GGUF's types stay in the format layer and the loading path. */
    @Test
    public void rule4_formatTypesStayInTheFormatLayerAndTheLoaders() {
        assertMatchesAllowlist(
                "Rule 4 (GGUF types outside the format layer and the loaders)",
                ArchRules.rule4FormatTypesOutsideFormatAndLoaders(classes()),
                Allowlists.RULE_4);
    }

    /**
     * Rule 4 — the runtime tensor vocabulary stays free of file-format and backend types.
     *
     * <p>Added with the package it watches. The runtime `DataType` exists so that the layers above
     * the format can speak about execution representations without naming `GGMLType`; a dependency
     * from here on the format or the backend would defeat the point before the mapping that needs
     * it is even written.
     */
    @Test
    public void rule4_theRuntimeTensorVocabularyNamesNoFormatOrBackendType() {
        for (String pkg : new String[] {ArchRules.RUNTIME_TENSOR, ArchRules.RUNTIME_METRICS}) {
            Set<String> violations =
                    ArchRules.rule4RuntimeNamesFormatOrBackendTypes(classes(), pkg);
            assertTrue(
                    "Rule 4 in "
                            + pkg
                            + " passes today and must stay passing; new violations: "
                            + violations,
                    violations.isEmpty());
        }
    }

    /**
     * Rule 17 — the metrics seam is written from below and read from above, which only works while
     * it depends on neither side. Added with the package, before anything writes to it, per the
     * "rules not yet enforceable" policy: the test comes first.
     */
    /**
     * Rule 3 — the program layer does not import TornadoVM or a backend.
     *
     * <p>No allowlist, and there must never be one. A program that imports a backend is not a
     * program.
     */
    @Test
    public void rule3_theProgramLayerDoesNotImportTornadoOrABackend() {
        Set<String> violations =
                ArchRules.rule3ProgramImportsBackend(
                        classes(), ArchRules.PROGRAM, ArchRules.TORNADO_BACKEND);
        assertTrue(
                "Rule 3: the program layer must not name TornadoVM or a backend; violations: "
                        + violations,
                violations.isEmpty());
    }

    /** Rule 14 — the program and runtime layers do not assume generation. */
    @Test
    public void rule14_coreAbstractionsDoNotAssumeGeneration() {
        Set<String> violations =
                ArchRules.rule14CoreAssumesGeneration(
                        classes(), ArchRules.PROGRAM, ArchRules.RUNTIME);
        assertTrue(
                "Rule 14: core abstractions must not require a tokenizer, chat format or"
                        + " sampler; violations: "
                        + violations,
                violations.isEmpty());
    }

    @Test
    public void rule17_theMetricsSeamDependsOnNothing() {
        Set<String> violations =
                ArchRules.rule17MetricsSeamDependsOnNothing(classes(), ArchRules.RUNTIME_METRICS);
        assertTrue(
                "Rule 17 passes today and must stay passing; new violations: " + violations,
                violations.isEmpty());
    }

    @Test
    public void rule8a_generationPolicyIsSeparateFromForwardExecution() {
        assertMatchesAllowlist(
                "Rule 8a (lower layers depending on generation policy)",
                ArchRules.rule8aLowerLayersDependOnGenerationPolicy(classes()),
                Allowlists.RULE_8A);
    }

    /**
     * Rule 8b — sampling is an operation and may execute on the device, so a device sampler is not
     * an 8a violation and must never be allowlisted as one. No such class exists yet; this guard
     * exists so that adding one cannot quietly go through the allowlist.
     */
    @Test
    public void rule8b_deviceSamplerIsNeverOnRule8aAllowlist() {
        for (String entry : Allowlists.RULE_8A) {
            assertFalse(
                    "on-device sampling is an operation (Rule 8b), not generation policy — "
                            + entry
                            + " must not be allowlisted under Rule 8a",
                    entry.startsWith(ArchRules.TORNADO_BACKEND + "."));
        }
    }

    @Test
    public void rule15_modelTypeDispatchStaysInTheProviders() {
        assertMatchesAllowlist(
                "Rule 15 (dispatch on ModelType outside the providers)",
                ArchRules.rule15ModelTypeDispatchOutsideProviders(classes()),
                Allowlists.RULE_15);
    }

    @Test
    public void rule16_noConsoleIoOutsideTheCliIntegration() {
        assertMatchesAllowlist(
                "Rule 16 (console I/O in library code)",
                ArchRules.rule16ConsoleIoOutsideCli(classes()),
                Allowlists.RULE_16);
    }

    @Test
    public void allowlistEntriesAreFullyQualifiedNames() {
        for (Set<String> list :
                Set.of(
                        Allowlists.RULE_1,
                        Allowlists.RULE_2,
                        Allowlists.RULE_5,
                        Allowlists.RULE_8A,
                        Allowlists.RULE_16)) {
            for (String entry : list) {
                assertTrue(
                        "wildcards are banned in allowlists: " + entry,
                        !entry.contains("*") && !entry.contains(".."));
                assertTrue(
                        "allowlist entries must be fully qualified: " + entry,
                        entry.startsWith(ProductionClasses.ROOT_PACKAGE + "."));
            }
        }
    }

    /** Loaded-model types only — loaders and builders are out of Rule 5's scope by its own text. */
    private static boolean isLoadedModelType(JavaClass c) {
        return !c.isInterface()
                        && c.getAllRawSuperclasses().stream()
                                .anyMatch(
                                        s ->
                                                s.getName()
                                                        .equals(
                                                                "org.beehive.jitllm.model.AbstractModel"))
                || c.getName().equals("org.beehive.jitllm.model.AbstractModel");
    }

    private static void assertMatchesAllowlist(
            String rule, Set<String> actual, Set<String> allowed) {
        Set<String> unlisted = new TreeSet<>(actual);
        unlisted.removeAll(allowed);
        Set<String> stale = new TreeSet<>(allowed);
        stale.removeAll(actual);

        if (unlisted.isEmpty() && stale.isEmpty()) {
            return;
        }
        StringBuilder sb = new StringBuilder(rule).append(" failed.\n");
        if (!unlisted.isEmpty()) {
            sb.append(
                    "  NEW violations (fix the code, or record a maintainer decision to add them):\n");
            unlisted.forEach(v -> sb.append("    ").append(v).append('\n'));
        }
        if (!stale.isEmpty()) {
            sb.append("  STALE allowlist entries (these no longer violate — delete them):\n");
            stale.forEach(v -> sb.append("    ").append(v).append('\n'));
        }
        fail(sb.toString());
    }
}
