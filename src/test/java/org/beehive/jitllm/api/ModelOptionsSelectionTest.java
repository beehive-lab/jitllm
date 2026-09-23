package org.beehive.jllm.api;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

import org.beehive.jllm.runtime.backend.BackendId;
import org.beehive.jllm.runtime.backend.DeviceCapability;
import org.beehive.jllm.runtime.backend.DeviceSelector;
import org.junit.Test;

public class ModelOptionsSelectionTest {

    @Test
    public void nothingStatedMeansTheExistingMechanismStillDecides() {
        assertNull(ModelOptions.defaults().backend());
        assertNull(ModelOptions.defaults().device());
    }

    @Test
    public void anExplicitBackendIsCarried() {
        ModelOptions options = ModelOptions.builder().backend(BackendId.CPU).build();
        assertEquals(BackendId.CPU, options.backend());
    }

    @Test
    public void aSelectorSuppliesTheBackendWhenNoneIsStatedDirectly() {
        ModelOptions options =
                ModelOptions.builder().device(DeviceSelector.backend(BackendId.METAL)).build();
        assertEquals(BackendId.METAL, options.resolvedBackend());
    }

    @Test
    public void anExplicitBackendAndAnAgreeingSelectorAreFine() {
        ModelOptions options =
                ModelOptions.builder()
                        .backend(BackendId.METAL)
                        .device(DeviceSelector.backend(BackendId.METAL))
                        .build();
        assertEquals(BackendId.METAL, options.resolvedBackend());
    }

    @Test
    public void contradictingTheSelectorIsRejectedAtBuild() {
        // Answering this with a precedence rule would pick one of the caller\u0027s two intentions
        // without telling them.
        IllegalArgumentException thrown =
                assertThrows(
                        IllegalArgumentException.class,
                        () ->
                                ModelOptions.builder()
                                        .backend(BackendId.CPU)
                                        .device(DeviceSelector.backend(BackendId.CUDA))
                                        .build());
        assertTrue(thrown.getMessage(), thrown.getMessage().contains("different backends"));
    }

    /** The selector can express more than the tree can honour, and the gap fails loudly. */
    @Test
    public void whatCannotBeHonouredThrowsRatherThanBeingIgnored() {
        assertThrows(
                UnsupportedOperationException.class,
                () ->
                        ModelOptions.builder()
                                .device(DeviceSelector.backend(BackendId.CUDA).withIndex(1))
                                .build());
        assertThrows(
                UnsupportedOperationException.class,
                () ->
                        ModelOptions.builder()
                                .device(DeviceSelector.any().withNameContaining("nvidia"))
                                .build());
        assertThrows(
                UnsupportedOperationException.class,
                () ->
                        ModelOptions.builder()
                                .device(
                                        DeviceSelector.any()
                                                .requiring(DeviceCapability.TENSOR_CORE_MMA))
                                .build());
    }

    // ── Metal parity, backlog task 4 ─────────────────────────────────────────
    //
    // Metal.BackendId is a value like any other (BackendId.of("metal")), so the cases above already
    // cover it structurally. These pin that no special-casing is needed for Metal specifically —
    // the
    // same "no platform branch in user code" claim task 4 requires, made permanent rather than
    // inferred from the generic cases using CPU/PTX/CUDA.

    @Test
    public void metalIsCarriedExactlyLikeAnyOtherBackend() {
        ModelOptions options = ModelOptions.builder().backend(BackendId.METAL).build();
        assertEquals(BackendId.METAL, options.backend());
        assertEquals(BackendId.METAL, options.resolvedBackend());
    }

    @Test
    public void aMetalSelectorSuppliesTheBackendWhenNoneIsStatedDirectly() {
        ModelOptions options =
                ModelOptions.builder().device(DeviceSelector.backend(BackendId.METAL)).build();
        assertEquals(BackendId.METAL, options.resolvedBackend());
    }

    @Test
    public void anExplicitMetalBackendAndAnAgreeingSelectorAreFine() {
        ModelOptions options =
                ModelOptions.builder()
                        .backend(BackendId.METAL)
                        .device(DeviceSelector.backend(BackendId.METAL))
                        .build();
        assertEquals(BackendId.METAL, options.resolvedBackend());
    }

    @Test
    public void contradictingTheSelectorIsRejectedForMetalToo() {
        IllegalArgumentException thrown =
                assertThrows(
                        IllegalArgumentException.class,
                        () ->
                                ModelOptions.builder()
                                        .backend(BackendId.CUDA)
                                        .device(DeviceSelector.backend(BackendId.METAL))
                                        .build());
        assertTrue(thrown.getMessage(), thrown.getMessage().contains("different backends"));
    }

    /**
     * The exact case task 4's acceptance names: an unhonourable Metal selector throws a named error
     * rather than silently selecting another backend or device. {@code DEVICE_SELECTOR_UNSUPPORTED}
     * / {@code CAPABILITY_UNAVAILABLE} are the named diagnostic codes {@code
     * rejectWhatCannotBeHonoured} already carries; this pins that Metal is rejected through the
     * same path as every other backend, not silently accepted and then dropped.
     */
    @Test
    public void whatCannotBeHonouredThrowsForMetalToo() {
        UnsupportedOperationException byIndex =
                assertThrows(
                        UnsupportedOperationException.class,
                        () ->
                                ModelOptions.builder()
                                        .device(
                                                DeviceSelector.backend(BackendId.METAL)
                                                        .withIndex(0))
                                        .build());
        assertTrue(byIndex.getMessage(), byIndex.getMessage().contains("GPUL-CFG-001"));

        UnsupportedOperationException byName =
                assertThrows(
                        UnsupportedOperationException.class,
                        () ->
                                ModelOptions.builder()
                                        .device(
                                                DeviceSelector.backend(BackendId.METAL)
                                                        .withNameContaining("apple"))
                                        .build());
        assertTrue(byName.getMessage(), byName.getMessage().contains("GPUL-CFG-001"));

        UnsupportedOperationException byCapability =
                assertThrows(
                        UnsupportedOperationException.class,
                        () ->
                                ModelOptions.builder()
                                        .device(
                                                DeviceSelector.backend(BackendId.METAL)
                                                        .requiring(
                                                                DeviceCapability
                                                                        .SPLIT_KV_ATTENTION))
                                        .build());
        assertTrue(byCapability.getMessage(), byCapability.getMessage().contains("GPUL-CFG-003"));
    }
}
