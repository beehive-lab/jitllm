package org.beehive.jllm.arch;

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import com.tngtech.archunit.core.domain.JavaClasses;
import com.tngtech.archunit.core.importer.ClassFileImporter;
import java.util.Set;
import org.junit.BeforeClass;
import org.junit.Test;

/**
 * A rule that only ever passes proves nothing — it could be matching the wrong package or silently
 * returning empty. These tests point each rule at {@code org.beehive.jllm.arch.fixture.model} and
 * assert it reports the planted violations.
 */
public class DependencyRulesSelfTest {

    private static final String FIXTURE = "org.beehive.jllm.arch.fixture.model";
    private static final String FIXTURE_BACKEND = "org.beehive.jllm.arch.fixture.backend";

    private static JavaClasses fixture;

    @BeforeClass
    public static void importFixture() {
        fixture = new ClassFileImporter().importPackages(FIXTURE);
        assertFalse("fixture classes did not import", fixture.isEmpty());
    }

    @Test
    public void rule1_flagsTornadoVmOutsideTheBackend() {
        Set<String> v = ArchRules.rule1TornadoVmOutsideBackend(fixture, FIXTURE_BACKEND);
        assertTrue(
                "Rule 1 did not flag the fixture: " + v, v.contains(FIXTURE + ".ViolatingModel"));
    }

    @Test
    public void rule2_flagsModelDependingOnTornado() {
        Set<String> v = ArchRules.rule2ModelDependsOnTornado(fixture, FIXTURE, FIXTURE_BACKEND);
        assertTrue(
                "Rule 2 did not flag the fixture: " + v, v.contains(FIXTURE + ".ViolatingModel"));
    }

    @Test
    public void rule18_flagsLowerTiersReachingTheEngine() {
        Set<String> v = ArchRules.rule18LowerTiersReachEngine(fixture, FIXTURE);
        assertTrue(
                "Rule 18 did not flag the fixture: " + v,
                v.contains(FIXTURE + ".ViolatingEngineUser"));
    }

    @Test
    public void rule5_flagsMutableModelFields() {
        Set<String> v =
                ArchRules.rule5MutableModelFields(
                        fixture, c -> c.getName().equals(FIXTURE + ".ViolatingModel"));
        assertTrue(
                "Rule 5 did not flag the mutable field: " + v,
                v.contains(FIXTURE + ".ViolatingModel"));
    }

    @Test
    public void m12db_flagsANeutralSpiTypeThatNamesAnImplementation() {
        String spi = "org.beehive.jllm.arch.fixture.spi";
        JavaClasses spiFixture = new ClassFileImporter().importPackages(spi);
        Set<String> v =
                ArchRules.m12dbNeutralSpiDependsOnAnImplementation(
                        spiFixture, spi, ArchRules.TORNADO_BACKEND);
        assertTrue(
                "the neutral-SPI rule did not flag the fixture: " + v,
                v.contains(spi + ".ViolatingNeutralSpi"));
    }

    @Test
    public void rule7_flagsKvStorageReachableFromModel() {
        Set<String> v = ArchRules.rule7ModelReachesKvStorage(fixture, FIXTURE);
        assertTrue(
                "Rule 7 did not flag the KV user: " + v, v.contains(FIXTURE + ".ViolatingKvUser"));
    }

    /**
     * Rule 4's fixture is the same one Rule 1 uses: a class naming TornadoVM types. Pointed at as
     * if it were a runtime package, it must be reported — otherwise the guard on {@code
     * runtime.tensor} is only ever passing because it is looking at the wrong package.
     */
    @Test
    public void rule4_flagsARuntimePackageThatNamesABackendType() {
        Set<String> v = ArchRules.rule4RuntimeNamesFormatOrBackendTypes(fixture, FIXTURE);
        assertTrue(
                "Rule 4 did not flag the fixture: " + v, v.contains(FIXTURE + ".ViolatingModel"));
    }

    /** The other half: a format type, which is the one Rule 4 is actually named for. */
    @Test
    public void rule4_flagsARuntimePackageThatNamesAFormatType() {
        String runtime = "org.beehive.jllm.arch.fixture.runtime";
        JavaClasses runtimeFixture = new ClassFileImporter().importPackages(runtime);
        Set<String> v = ArchRules.rule4RuntimeNamesFormatOrBackendTypes(runtimeFixture, runtime);
        assertTrue(
                "Rule 4 did not flag the format-naming fixture: " + v,
                v.contains(runtime + ".ViolatingRuntimeVocabulary"));
    }

    @Test
    public void rule17_flagsAMetricsSeamThatReachesUpwards() {
        String seam = "org.beehive.jllm.arch.fixture.metrics";
        JavaClasses seamFixture = new ClassFileImporter().importPackages(seam);
        Set<String> v = ArchRules.rule17MetricsSeamDependsOnNothing(seamFixture, seam);
        assertTrue("Rule 17 did not flag the fixture: " + v, v.contains(seam + ".ViolatingSeam"));
    }

    @Test
    public void rule11_flagsPlanTypesOutsideTheBackend() {
        Set<String> v = ArchRules.rule11PlanTypesOutsideBackend(fixture, FIXTURE_BACKEND);
        assertTrue(
                "Rule 11 did not flag TaskGraph use: " + v,
                v.contains(FIXTURE + ".ViolatingModel"));
    }

    @Test
    public void rule8a_flagsLowerLayerDependingOnGenerationPolicy() {
        Set<String> v = ArchRules.rule8aLowerLayersDependOnGenerationPolicy(fixture);
        assertTrue(
                "Rule 8a did not flag the CLI dependency: " + v,
                v.contains(FIXTURE + ".ViolatingConsoleAndPolicyUser"));
    }

    @Test
    public void rule16_flagsConsoleIoInLibraryCode() {
        Set<String> v = ArchRules.rule16ConsoleIoOutsideCli(fixture);
        assertTrue(
                "Rule 16 did not flag the console I/O: " + v,
                v.contains(FIXTURE + ".ViolatingConsoleAndPolicyUser"));
    }

    @Test
    public void rulesAreQuietWhenTheFixtureIsTreatedAsTheBackend() {
        // Same classes, but now inside the backend package: Rules 1 and 11 must go silent.
        assertTrue(ArchRules.rule1TornadoVmOutsideBackend(fixture, FIXTURE).isEmpty());
        assertTrue(ArchRules.rule11PlanTypesOutsideBackend(fixture, FIXTURE).isEmpty());
    }

    @Test
    public void rule3_flagsAProgramTypeNamingTornado() {
        String program = "org.beehive.jllm.arch.fixture.program";
        JavaClasses f = new ClassFileImporter().importPackages(program);
        Set<String> v = ArchRules.rule3ProgramImportsBackend(f, program, ArchRules.TORNADO_BACKEND);
        assertTrue(
                "Rule 3 did not flag the fixture: " + v,
                v.contains(program + ".ViolatingProgramComponent"));
    }

    /** Rule 14's fixture requires a tokenizer to be constructed, which a core type may not. */
    @Test
    public void rule14_flagsACoreTypeRequiringATokenizer() {
        String program = "org.beehive.jllm.arch.fixture.program";
        JavaClasses f = new ClassFileImporter().importPackages(program);
        Set<String> v = ArchRules.rule14CoreAssumesGeneration(f, program);
        assertTrue(
                "Rule 14 did not flag the fixture: " + v,
                v.contains(program + ".ViolatingGenerationAssumption"));
    }
}
