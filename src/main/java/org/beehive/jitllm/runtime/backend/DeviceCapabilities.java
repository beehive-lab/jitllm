package org.beehive.jllm.runtime.backend;

import java.util.Collection;
import java.util.Objects;
import java.util.Set;
import java.util.TreeSet;
import java.util.stream.Collectors;
import org.beehive.jllm.api.Experimental;

/**
 * What a device can do that lowering may vary on — an immutable set, and a cache-key component.
 *
 * <h2>Why this is separate from {@link DeviceId}</h2>
 *
 * <p>It would be safe to leave out of the compiled-program cache key <b>only</b> if capabilities
 * were a deterministic, immutable function of the device identifier. They are not: the scheduler
 * mode is overridable, so two lowerings of one program on one device can differ. Two different
 * lowerings must never collide because their device identifier happened to match.
 *
 * <p>The general rule, which is also the bar for adding a {@link DeviceCapability}: <b>if changing
 * an input can change task count, task names, kernels, grid entries or bindings, it must
 * distinguish cache entries.</b>
 */
@Experimental
public final class DeviceCapabilities {

    /** A device that supports nothing optional — the honest answer for a plain CPU. */
    public static final DeviceCapabilities NONE = of(Set.of());

    private final Set<DeviceCapability> capabilities;
    private final String fingerprint;

    private DeviceCapabilities(Set<DeviceCapability> capabilities) {
        this.capabilities = capabilities;
        // Sorted by name so the fingerprint does not depend on iteration order: a cache key that
        // varies with a HashSet's layout would be a cache key that varies for no reason.
        this.fingerprint =
                capabilities.stream()
                        .map(DeviceCapability::name)
                        .collect(Collectors.toCollection(TreeSet::new))
                        .stream()
                        .collect(Collectors.joining(","));
    }

    public static DeviceCapabilities of(Collection<DeviceCapability> capabilities) {
        Objects.requireNonNull(capabilities, "capabilities");
        return new DeviceCapabilities(Set.copyOf(capabilities));
    }

    public static DeviceCapabilities of(DeviceCapability... capabilities) {
        return of(Set.of(capabilities));
    }

    public boolean supports(DeviceCapability capability) {
        return capabilities.contains(capability);
    }

    public boolean supportsAll(Collection<DeviceCapability> required) {
        return capabilities.containsAll(required);
    }

    public Set<DeviceCapability> asSet() {
        return capabilities;
    }

    /** A canonical fingerprint for the cache key. Stable across runs, and readable in a log. */
    public String fingerprint() {
        return fingerprint.isEmpty() ? "none" : fingerprint;
    }

    @Override
    public boolean equals(Object other) {
        return other instanceof DeviceCapabilities that && capabilities.equals(that.capabilities);
    }

    @Override
    public int hashCode() {
        return capabilities.hashCode();
    }

    @Override
    public String toString() {
        return "capabilities=" + fingerprint();
    }
}
