package org.beehive.jllm.backend.tornado.lowering;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import java.util.ArrayList;
import java.util.List;
import org.beehive.jllm.model.architecture.LlamaProgramDescription;
import org.beehive.jllm.model.llama.LlamaConfiguration;
import org.beehive.jllm.program.InferenceProgram;
import org.beehive.jllm.program.PhaseId;
import org.beehive.jllm.program.ProgramComponent;
import org.beehive.jllm.program.ProgramSignature;
import org.beehive.jllm.program.op.OperationKind;
import org.beehive.jllm.runtime.backend.BackendId;
import org.beehive.jllm.runtime.backend.CompileOptions;
import org.beehive.jllm.runtime.backend.DeviceCapabilities;
import org.beehive.jllm.runtime.backend.DeviceCapability;
import org.beehive.jllm.runtime.backend.DeviceId;
import org.beehive.jllm.runtime.model.ArchitectureId;
import org.beehive.jllm.runtime.tensor.DataType;
import org.junit.Test;

/**
 * These are the checks that make this a lowering rather than a wrapper. A class that matched on the
 * architecture string and called the legacy builder regardless of the components it was handed
 * would pass a "does it produce a plan" test and fail every one of these.
 */
public class LlamaLoweringTest {

    /** A fixed backend and device identity: these cases are about what the key distinguishes. */
    private static final BackendId BACKEND = BackendId.of("tornado");

    private static final DeviceId DEVICE = DeviceId.of(BACKEND, "gpu0");

    private static final LlamaLowering LOWERING =
            new LlamaLowering(
                    new CompileOptions(false),
                    DeviceCapabilities.of(DeviceCapability.SINGLE_PASS_RMS));

    private static final int DIM = 64;
    private static final int LAYERS = 2;

    @Test
    public void theDescribedProgramIsTheSupportedTopology() {
        InferenceProgram program = program(false);
        LOWERING.validate(program); // must not throw
        assertTrue(LOWERING.supports(program));

        assertEquals(
                "embedding, two layers, final norm, projection",
                1 + LAYERS + 2,
                program.components().size());
        assertEquals(
                "prefill stops before the final norm and the projection",
                1 + LAYERS,
                program.componentsFor(PhaseId.PREFILL).size());
        assertEquals(
                "decode runs everything",
                1 + LAYERS + 2,
                program.componentsFor(PhaseId.DECODE).size());
    }

    /** Device-resident sampling is policy: it adds a component and changes the signature. */
    @Test
    public void deviceSamplingAddsAComponentAndChangesTheSignature() {
        InferenceProgram host = program(false);
        InferenceProgram device = program(true);

        assertEquals(host.components().size() + 1, device.components().size());
        assertEquals(
                OperationKind.ARG_MAX,
                ((ProgramComponent.Leaf) device.components().getLast()).operation().kind());
        assertNotEquals(
                "sampling residency is part of the policy, so of the signature",
                host.signature(),
                device.signature());
        LOWERING.validate(device);
    }

    /**
     * The correction that matters most: the FP16 key/value cache leaves the <b>component
     * sequence</b> identical and changes the <b>signature</b>, because the fixed key/value bindings
     * have a different dtype. Same logical operations, different cache entry.
     */
    @Test
    public void theKeyValueRepresentationChangesTheSignatureButNotTheSequence() {
        InferenceProgram f32 =
                LlamaProgramDescription.build(config(), DataType.F16, DataType.F32, false, false);
        InferenceProgram f16 =
                LlamaProgramDescription.build(config(), DataType.F16, DataType.F16, false, false);

        assertEquals("the logical sequence is the same", kinds(f32), kinds(f16));
        assertNotEquals(
                "but the key/value bindings differ, so the signature does",
                f32.signature(),
                f16.signature());

        assertEquals(DataType.F32, kvBindingType(f32.signature()));
        assertEquals(DataType.F16, kvBindingType(f16.signature()));
    }

    @Test
    public void theWeightRepresentationChangesTheSignatureButNotTheSequence() {
        InferenceProgram f16 =
                LlamaProgramDescription.build(config(), DataType.F16, DataType.F32, false, false);
        InferenceProgram q8 =
                LlamaProgramDescription.build(config(), DataType.Q8_0, DataType.F32, false, false);

        assertEquals("the logical sequence is the same", kinds(f16), kinds(q8));
        assertNotEquals(
                "an F16 and a Q8_0 Llama are not the same program",
                f16.signature(),
                q8.signature());

        LOWERING.validate(q8); // must not throw
        assertTrue(LOWERING.supports(q8));
    }

    /**
     * A description that mixed them would name a program this backend has no task graphs for.
     * Lowering it would take whichever representation the model happened to carry, and the
     * description would have decided nothing — the failure mode this whole slice exists to remove.
     */
    @Test
    public void aMixedWeightRepresentationIsRefused() {
        InferenceProgram original =
                LlamaProgramDescription.build(config(), DataType.Q8_0, DataType.F32, false, false);
        ProgramComponent.Composite layer =
                (ProgramComponent.Composite) original.components().get(1);
        List<ProgramComponent> mixed = new ArrayList<>(layer.children());
        ProgramComponent.Leaf q = (ProgramComponent.Leaf) layer.children().get(1);
        org.beehive.jllm.program.op.MatVec original_ =
                (org.beehive.jllm.program.op.MatVec) q.operation();
        mixed.set(
                1,
                new ProgramComponent.Leaf(
                        q.name(),
                        new org.beehive.jllm.program.op.MatVec(
                                original_.weight(),
                                original_.input(),
                                original_.output(),
                                original_.rows(),
                                original_.columns(),
                                DataType.F16),
                        q.phases()));

        assertRefused(
                "one layer's projection at another representation",
                () ->
                        LOWERING.validate(
                                rebuild(
                                        original,
                                        1,
                                        new ProgramComponent.Composite(
                                                layer.name(), mixed, layer.phases()))));
    }

    @Test
    public void anUnsupportedWeightRepresentationIsRefused() {
        for (DataType unsupported : List.of(DataType.F32, DataType.BF16, DataType.Q4_0)) {
            InferenceProgram program =
                    LlamaProgramDescription.build(
                            config(), unsupported, DataType.F32, false, false);
            assertFalse(
                    unsupported + " has no single-token task graphs on this backend",
                    LOWERING.supports(program));
        }
    }

    /**
     * Mistral's forward pass is {@code InferenceCore.forwardJava} — Llama's own method — so a
     * second description would be a copy asserting what the code already says. What must not follow
     * is that they become one program: the architecture is in the signature, and a lowering
     * constructed for one family refuses the other's program.
     */
    @Test
    public void mistralIsTheSameShapeButNotTheSameProgram() {
        InferenceProgram llama =
                LlamaProgramDescription.build(config(), DataType.F16, DataType.F32, false, false);
        InferenceProgram mistral =
                LlamaProgramDescription.build(
                        ArchitectureId.of("mistral"),
                        config(),
                        DataType.F16,
                        DataType.F32,
                        false,
                        false);

        assertEquals("the same operations in the same order", kinds(llama), kinds(mistral));
        assertNotEquals(
                "but not the same program — the architecture is in the signature",
                llama.signature(),
                mistral.signature());

        LlamaLowering mistralLowering =
                new LlamaLowering(
                        new CompileOptions(false),
                        DeviceCapabilities.of(DeviceCapability.SINGLE_PASS_RMS),
                        ArchitectureId.of("mistral"));
        mistralLowering.validate(mistral);
        assertFalse(
                "a lowering built for Mistral must refuse a Llama program",
                mistralLowering.supports(llama));
        assertFalse("and the Llama one must refuse Mistral's", LOWERING.supports(mistral));
    }

    /** A reordered layer is refused, naming what was expected. */
    @Test
    public void aReorderedLayerIsRefused() {
        InferenceProgram original = program(false);
        ProgramComponent.Composite layer =
                (ProgramComponent.Composite) original.components().get(1);
        List<ProgramComponent> swapped = new ArrayList<>(layer.children());
        swapped.set(0, layer.children().get(1));
        swapped.set(1, layer.children().get(0));

        assertRefused(
                "a reordered layer",
                () ->
                        LOWERING.validate(
                                rebuild(
                                        original,
                                        1,
                                        new ProgramComponent.Composite(
                                                layer.name(), swapped, layer.phases()))));
    }

    /** So is a layer missing a component. */
    @Test
    public void aTruncatedLayerIsRefused() {
        InferenceProgram original = program(false);
        ProgramComponent.Composite layer =
                (ProgramComponent.Composite) original.components().get(1);
        List<ProgramComponent> short_ = new ArrayList<>(layer.children());
        short_.removeLast();

        assertRefused(
                "a layer missing its final residual add",
                () ->
                        LOWERING.validate(
                                rebuild(
                                        original,
                                        1,
                                        new ProgramComponent.Composite(
                                                layer.name(), short_, layer.phases()))));
    }

    /** And a program for another architecture, even though nothing else about it differs. */
    @Test
    public void anotherArchitectureIsRefused() {
        InferenceProgram original = program(false);
        ProgramSignature renamed =
                new ProgramSignature(
                        ArchitectureId.of("qwen3"),
                        original.signature().policy(),
                        original.signature().capacity(),
                        original.signature().components(),
                        original.signature().phases(),
                        original.signature().bindings());
        assertRefused(
                "another architecture", () -> LOWERING.validate(InferenceProgram.of(renamed)));
        assertFalse(LOWERING.supports(InferenceProgram.of(renamed)));
    }

    /**
     * Device capabilities are in the cache key because they are <b>not</b> a function of the device
     * identifier — the scheduler mode is overridable, so two lowerings of one program on one device
     * can differ. Without this, they would collide.
     */
    @Test
    public void deviceCapabilitiesDistinguishCacheEntries() {
        ProgramSignature signature = program(false).signature();
        BindingDomain domain = BindingDomain.create("test");
        CompileOptions options = new CompileOptions(false);

        ProgramCacheKey nvidia =
                ProgramCacheKey.of(
                        signature,
                        BACKEND,
                        DEVICE,
                        options,
                        DeviceCapabilities.of(DeviceCapability.SINGLE_PASS_RMS),
                        domain);
        ProgramCacheKey other =
                ProgramCacheKey.of(
                        signature, BACKEND, DEVICE, options, DeviceCapabilities.NONE, domain);

        assertNotEquals(
                "the same device with a different scheduler emits a different task set",
                nvidia,
                other);
    }

    /**
     * The typed key is where a "simplification" could do real damage without failing anything else
     * — passing a {@code DeviceSelector} instead of a {@code DeviceId} would compile a device's
     * programs once per way of asking for it, and merging the backend into the device handle would
     * let two backends' identically-numbered devices share entries.
     */
    @Test
    public void theBackendAndTheDeviceIdentityDistinguishCacheEntries() {
        ProgramSignature signature = program(false).signature();
        BindingDomain domain = BindingDomain.create("test");
        CompileOptions options = new CompileOptions(false);
        DeviceCapabilities capabilities = DeviceCapabilities.of(DeviceCapability.SINGLE_PASS_RMS);

        ProgramCacheKey here =
                ProgramCacheKey.of(signature, BACKEND, DEVICE, options, capabilities, domain);

        assertEquals(
                "the same backend, device, options, capabilities and domain is one entry",
                here,
                ProgramCacheKey.of(
                        signature,
                        BackendId.of("tornado"),
                        DeviceId.of(BackendId.of("tornado"), "gpu0"),
                        options,
                        capabilities,
                        domain));

        assertNotEquals(
                "a second device on the same backend is a second entry",
                here,
                ProgramCacheKey.of(
                        signature,
                        BACKEND,
                        DeviceId.of(BACKEND, "gpu1"),
                        options,
                        capabilities,
                        domain));

        // The same handle on two backends is two devices, which is why the backend is inside
        // DeviceId as well as beside it.
        BackendId other = BackendId.of("other-backend");
        assertNotEquals(
                "one handle on two backends is two entries",
                here,
                ProgramCacheKey.of(
                        signature,
                        other,
                        DeviceId.of(other, "gpu0"),
                        options,
                        capabilities,
                        domain));
    }

    /** Compile options too — the same computation, a different compiled artefact. */
    @Test
    public void compileOptionsDistinguishCacheEntries() {
        ProgramSignature signature = program(false).signature();
        BindingDomain domain = BindingDomain.create("test");
        DeviceCapabilities capabilities = DeviceCapabilities.of(DeviceCapability.SINGLE_PASS_RMS);

        assertNotEquals(
                ProgramCacheKey.of(
                        signature, BACKEND, DEVICE, new CompileOptions(true), capabilities, domain),
                ProgramCacheKey.of(
                        signature,
                        BACKEND,
                        DEVICE,
                        new CompileOptions(false),
                        capabilities,
                        domain));
    }

    /** The binding domain compares by identity, not by description. Two domains are two entries. */
    @Test
    public void theBindingDomainComparesByIdentity() {
        ProgramSignature signature = program(false).signature();
        CompileOptions options = new CompileOptions(false);
        DeviceCapabilities capabilities = DeviceCapabilities.of(DeviceCapability.SINGLE_PASS_RMS);

        BindingDomain one = BindingDomain.create("same-label");
        BindingDomain two = BindingDomain.create("same-label");

        assertEquals(
                "one domain is one entry",
                ProgramCacheKey.of(signature, BACKEND, DEVICE, options, capabilities, one),
                ProgramCacheKey.of(signature, BACKEND, DEVICE, options, capabilities, one));
        assertNotEquals(
                "identical labels are still two domains",
                ProgramCacheKey.of(signature, BACKEND, DEVICE, options, capabilities, one),
                ProgramCacheKey.of(signature, BACKEND, DEVICE, options, capabilities, two));
    }

    /** Concurrent identical misses compile once; a failure is not cached. */
    @Test
    public void theCacheCompilesOnceAndDoesNotCacheFailure() {
        CompiledProgramCache cache = new CompiledProgramCache();
        ProgramCacheKey key =
                ProgramCacheKey.of(
                        program(false).signature(),
                        BACKEND,
                        DEVICE,
                        new CompileOptions(false),
                        DeviceCapabilities.of(DeviceCapability.SINGLE_PASS_RMS),
                        BindingDomain.create("test"));

        try {
            cache.acquire(
                    key,
                    () -> {
                        throw new IllegalStateException("compilation failed");
                    });
            fail("the failure must propagate");
        } catch (IllegalStateException expected) {
            // the contract
        }
        assertEquals("a failed compilation leaves no entry behind", 0, cache.size());
    }

    /**
     * The central question this slice exists to answer: <b>do two standalone sessions share one
     * compiled program?</b>
     *
     * <p>So two sessions occupy two domains and get two entries. A hit across them would hand one
     * session a program bound to the other's buffers, which under CUDA graph capture produces
     * <b>wrong output rather than an error</b> (capability C1).
     *
     * <p>This test pins both halves of that: distinct domains do not share, and a shared domain
     * does. The second half is what will start being reachable when the workspace moves behind the
     * domain — the work that has to land before the weight duplication can disappear.
     */
    @Test
    public void twoDomainsDoNotShareAProgramAndOneDomainDoes() {
        CompiledProgramCache cache = new CompiledProgramCache();
        ProgramSignature signature = program(false).signature();
        CompileOptions options = new CompileOptions(false);
        DeviceCapabilities capabilities = DeviceCapabilities.of(DeviceCapability.SINGLE_PASS_RMS);

        BindingDomain sessionA = BindingDomain.create("sessionA");
        BindingDomain sessionB = BindingDomain.create("sessionB");

        StubPlan planA = new StubPlan();
        StubPlan planB = new StubPlan();
        StubPlan unused = new StubPlan();

        // Two sessions, two workspaces, two domains: two compiled programs.
        assertEquals(planA, acquire(cache, signature, options, capabilities, sessionA, planA));
        assertEquals(planB, acquire(cache, signature, options, capabilities, sessionB, planB));
        assertEquals("distinct domains must not share a compiled program", 2, cache.size());

        // The same domain asked twice compiles once — the mechanism sharing will use once the
        // workspace moves behind the domain.
        assertEquals(
                "a second acquisition in one domain must hit",
                planA,
                acquire(cache, signature, options, capabilities, sessionA, unused));
        assertEquals("and must not add an entry", 2, cache.size());
    }

    private static org.beehive.jllm.backend.tornado.TornadoVMMasterPlan acquire(
            CompiledProgramCache cache,
            ProgramSignature signature,
            CompileOptions options,
            DeviceCapabilities capabilities,
            BindingDomain domain,
            StubPlan plan) {
        ProgramCacheKey key =
                ProgramCacheKey.of(signature, BACKEND, DEVICE, options, capabilities, domain);
        return cache.acquire(key, () -> plan);
    }

    /** A stand-in for a compiled plan, so the cache's behaviour can be tested without a device. */
    private static class StubPlan implements org.beehive.jllm.backend.tornado.TornadoVMMasterPlan {
        @Override
        public uk.ac.manchester.tornado.api.TornadoExecutionPlan createExecutionPlan() {
            throw new UnsupportedOperationException();
        }

        @Override
        public void resetSequenceState() {}

        @Override
        public void forceCopyInReadOnlyData() {}

        @Override
        public uk.ac.manchester.tornado.api.types.arrays.FloatArray tornadoVMForwardDecode(int p) {
            throw new UnsupportedOperationException();
        }

        @Override
        public void freeTornadoExecutionPlan() {}
    }

    /**
     * Two boundaries share one carrier and one lock, which is what two sessions in one binding
     * domain will be. A boundary that returned a view of the carrier would show session B's values
     * in session A's result — the data race the copy-out exists to prevent.
     */
    @Test
    public void oneSessionsResultSurvivesAnotherSessionsInvocation() {
        uk.ac.manchester.tornado.api.types.arrays.FloatArray sharedCarrier =
                new uk.ac.manchester.tornado.api.types.arrays.FloatArray(4);
        uk.ac.manchester.tornado.api.types.arrays.IntArray control =
                new uk.ac.manchester.tornado.api.types.arrays.IntArray(2);
        java.util.concurrent.atomic.AtomicInteger round =
                new java.util.concurrent.atomic.AtomicInteger();

        StubPlan shared =
                new StubPlan() {
                    @Override
                    public uk.ac.manchester.tornado.api.types.arrays.FloatArray
                            tornadoVMForwardDecode(int p) {
                        float value = round.incrementAndGet();
                        for (int i = 0; i < sharedCarrier.getSize(); i++) {
                            sharedCarrier.set(i, value);
                        }
                        return sharedCarrier;
                    }
                };

        java.util.concurrent.atomic.AtomicInteger staged =
                new java.util.concurrent.atomic.AtomicInteger(-1);
        Object domainLock = new Object();
        SharedWorkspacePlan sessionA =
                new SharedWorkspacePlan(
                        shared, domainLock, control, 3, 4, staged::set, false, null);
        SharedWorkspacePlan sessionB =
                new SharedWorkspacePlan(
                        shared, domainLock, control, 7, 4, staged::set, false, null);

        InvocationBoundary.Result a = sessionA.invoke(11, 0);
        assertEquals("the boundary stages the token, not the caller", 11, staged.get());
        assertEquals("this session's slot is written into the control array", 3, control.get(1));
        assertEquals(1.0f, a.logits().get(0), 0f);

        InvocationBoundary.Result b = sessionB.invoke(12, 0);
        assertEquals(
                "the other session's slot goes into the same control array", 7, control.get(1));
        assertEquals(2.0f, b.logits().get(0), 0f);

        assertEquals(
                "session A's result must be untouched by session B's invocation",
                1.0f,
                a.logits().get(0),
                0f);
        for (int i = 0; i < sharedCarrier.getSize(); i++) {
            sharedCarrier.set(i, 99.0f);
        }
        assertEquals(
                "session A's result must not be a view of the shared carrier",
                1.0f,
                a.logits().get(0),
                0f);
        assertEquals(
                "session B's result must not be a view of the shared carrier",
                2.0f,
                b.logits().get(0),
                0f);
        assertTrue(
                "the two sessions must not share a result buffer either", a.logits() != b.logits());
    }

    /** A lowered result with no device sampling leaves the choice to the host sampler. */
    @Test
    public void aHostSampledResultReportsNoDeviceToken() {
        InvocationBoundary.Result result =
                new InvocationBoundary.Result(
                        org.beehive.jllm.backend.tornado.workspace.TornadoLogits.of(
                                new uk.ac.manchester.tornado.api.types.arrays.FloatArray(1)),
                        -1);
        assertTrue("a negative token means the host samples", !result.hasSampledToken());
    }

    /**
     * Two keys built from the same model and the same domain must be equal and hash equally.
     *
     * <p>The sweep found {@code compiledProgramCount()} of 2 where it should have been 1, so the
     * question is whether the key is unstable. If this passes, the instability is not in the key
     * and the second entry comes from somewhere else.
     */
    @Test
    public void theCacheKeyIsStableAcrossCalls() {
        BindingDomain domain = BindingDomain.create("one");
        CompileOptions options = new CompileOptions(false);
        DeviceCapabilities capabilities = DeviceCapabilities.of(DeviceCapability.SINGLE_PASS_RMS);

        ProgramCacheKey first =
                ProgramCacheKey.of(
                        program(false).signature(), BACKEND, DEVICE, options, capabilities, domain);
        ProgramCacheKey second =
                ProgramCacheKey.of(
                        program(false).signature(), BACKEND, DEVICE, options, capabilities, domain);

        assertEquals(
                "two keys for the same program in the same domain must be equal", first, second);
        assertEquals(
                "and must hash equally, or a HashMap holds two entries",
                first.hashCode(),
                second.hashCode());

        java.util.Map<ProgramCacheKey, String> map = new java.util.HashMap<>();
        map.put(first, "a");
        map.put(second, "b");
        assertEquals("one key, one entry", 1, map.size());
    }

    // helpers

    private static InferenceProgram program(boolean deviceSample) {
        return LlamaProgramDescription.build(
                config(), DataType.F16, DataType.F32, deviceSample, false);
    }

    private static LlamaConfiguration config() {
        return new LlamaConfiguration("FP16", DIM, 128, LAYERS, 4, 2, 48, 32, 1e-5f, 500000f);
    }

    private static InferenceProgram rebuild(
            InferenceProgram original, int index, ProgramComponent replacement) {
        List<ProgramComponent> components = new ArrayList<>(original.components());
        components.set(index, replacement);
        return InferenceProgram.of(
                new ProgramSignature(
                        original.signature().architecture(),
                        original.signature().policy(),
                        original.signature().capacity(),
                        components,
                        original.signature().phases(),
                        original.signature().bindings()));
    }

    private static List<String> kinds(InferenceProgram program) {
        List<String> out = new ArrayList<>();
        for (ProgramComponent component : program.components()) {
            if (component instanceof ProgramComponent.Leaf leaf) {
                out.add(leaf.operation().kind().name());
            } else {
                for (ProgramComponent child : ((ProgramComponent.Composite) component).children()) {
                    out.add(((ProgramComponent.Leaf) child).operation().kind().name());
                }
            }
        }
        return out;
    }

    private static DataType kvBindingType(ProgramSignature signature) {
        return signature.programFixed().stream()
                .filter(b -> b.role() == org.beehive.jllm.program.BindingRole.KV_POOL)
                .findFirst()
                .orElseThrow()
                .dataType();
    }

    private static void assertRefused(String what, Runnable validation) {
        try {
            validation.run();
            fail("must be refused: " + what);
        } catch (UnsupportedProgramException expected) {
            assertTrue(
                    "the refusal must say what it expected: " + expected.getMessage(),
                    expected.getMessage().contains("expected"));
        }
    }
}
