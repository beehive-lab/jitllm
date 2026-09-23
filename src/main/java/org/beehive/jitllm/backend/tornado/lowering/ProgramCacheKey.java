package org.beehive.jitllm.backend.tornado.lowering;

import java.util.Objects;
import org.beehive.jitllm.program.ProgramSignature;
import org.beehive.jitllm.runtime.backend.BackendId;
import org.beehive.jitllm.runtime.backend.CompileOptions;
import org.beehive.jitllm.runtime.backend.DeviceCapabilities;
import org.beehive.jitllm.runtime.backend.DeviceId;

/**
 * What makes two compiled programs the same one.
 *
 * <p>Five components, and each earns its place by being able to change the compiled artefact:
 *
 * <ul>
 *   <li>the {@link ProgramSignature} — architecture, policy, capacity, components, phases and the
 *       whole binding surface, including each weight's and each key/value array's representation;
 *   <li>the <b>backend</b> and the <b>device's identity</b> — a {@link BackendId} and a {@link
 *   <li>the {@link CompileOptions};
 *   <li>the {@link DeviceCapabilities} fingerprint — because capabilities are <b>not</b> a
 *       deterministic function of the device identifier, so two lowerings on one device can differ;
 *   <li>the {@link BindingDomain}, compared by <b>identity</b>.
 * </ul>
 * <p>The general rule the capability component comes from: <b>if changing an input can change task
 * count, task names, kernels, grid entries or bindings, it must distinguish cache entries.</b>
 *
 * <p>Note what follows for the FP16 key/value cache. It does not change the logical component
 * sequence, so it is tempting to call it a lowering choice — but it changes the <b>dtype and layout
 * of the fixed key/value bindings</b>, so it changes the signature, so it is a different entry
 * regardless of what the lowering does with it.
 */
public record ProgramCacheKey(
        ProgramSignature signature,
        BackendId backend,
        DeviceId device,
        CompileOptions compileOptions,
        String capabilityFingerprint,
        BindingDomain bindingDomain) {

    public ProgramCacheKey {
        Objects.requireNonNull(signature, "signature");
        Objects.requireNonNull(backend, "backend");
        Objects.requireNonNull(device, "device");
        Objects.requireNonNull(compileOptions, "compileOptions");
        Objects.requireNonNull(capabilityFingerprint, "capabilityFingerprint");
        Objects.requireNonNull(bindingDomain, "bindingDomain");
    }

    public static ProgramCacheKey of(
            ProgramSignature signature,
            BackendId backend,
            DeviceId device,
            CompileOptions compileOptions,
            DeviceCapabilities capabilities,
            BindingDomain domain) {
        return new ProgramCacheKey(
                signature, backend, device, compileOptions, capabilities.fingerprint(), domain);
    }

    /**
     * Identity comparison on the binding domain, value comparison on everything else.
     *
     * <p>Written out rather than left to the record's generated {@code equals} because the domain
     * must <b>not</b> compare by description: that is the whole point of it being in the key.
     * {@link BindingDomain} has no {@code equals} of its own, so the generated one would already do
     * this — spelling it out keeps a later addition of {@code equals} there from silently changing
     * what a cache hit means.
     */
    @Override
    public boolean equals(Object other) {
        if (this == other) {
            return true;
        }
        if (!(other instanceof ProgramCacheKey key)) {
            return false;
        }
        return bindingDomain == key.bindingDomain
                && signature.equals(key.signature)
                && backend.equals(key.backend)
                && device.equals(key.device)
                && compileOptions.equals(key.compileOptions)
                && capabilityFingerprint.equals(key.capabilityFingerprint);
    }

    @Override
    public int hashCode() {
        int result = signature.hashCode();
        result = 31 * result + backend.hashCode();
        result = 31 * result + device.hashCode();
        result = 31 * result + compileOptions.hashCode();
        result = 31 * result + capabilityFingerprint.hashCode();
        result = 31 * result + System.identityHashCode(bindingDomain);
        return result;
    }
}
