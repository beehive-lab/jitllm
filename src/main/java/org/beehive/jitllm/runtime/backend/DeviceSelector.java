package org.beehive.jllm.runtime.backend;

import java.util.Collection;
import java.util.Locale;
import java.util.Objects;
import java.util.Optional;
import java.util.OptionalInt;
import java.util.Set;
import org.beehive.jllm.api.Experimental;

/**
 * What is being <b>asked for</b> — a structured request, resolved by a backend into a {@link
 * Device}.
 *
 * <p>Every field is optional, and an empty selector means "whatever this backend would pick". That
 * is the current behaviour named rather than implied: everything in the tree resolves {@code
 * getBackend(0).getDefaultDevice()} today.
 *
 * <p><b>A selector is not a cache key.</b> Two selectors can name one device — an index and a name
 * fragment, say — and keying compiled programs on the request rather than on the resolved {@link
 * DeviceId} would compile that device's programs twice. This class therefore has value equality for
 * the convenience of tests and option objects, and is <b>never</b> a component of {@code
 * ProgramCacheKey}.
 *
 * <p>{@link #capabilityRequirements()} is a filter applied during resolution — "give me a device
 * that can do MMA" — and is not part of the resolved device's identity either.
 */
@Experimental
public final class DeviceSelector {

    private static final DeviceSelector ANY = new DeviceSelector(null, null, null, Set.of());

    private final BackendId backend;
    private final Integer index;
    private final String nameContains;
    private final Set<DeviceCapability> capabilityRequirements;

    private DeviceSelector(
            BackendId backend,
            Integer index,
            String nameContains,
            Set<DeviceCapability> capabilityRequirements) {
        this.backend = backend;
        this.index = index;
        this.nameContains = nameContains;
        this.capabilityRequirements = capabilityRequirements;
    }

    /** No constraint: the backend picks. */
    public static DeviceSelector any() {
        return ANY;
    }

    public static DeviceSelector backend(BackendId backend) {
        return any().withBackend(backend);
    }

    public DeviceSelector withBackend(BackendId backend) {
        Objects.requireNonNull(backend, "backend");
        return new DeviceSelector(backend, index, nameContains, capabilityRequirements);
    }

    /**
     * @param index the backend's own device ordinal
     */
    public DeviceSelector withIndex(int index) {
        if (index < 0) {
            throw new IllegalArgumentException("a device index must not be negative, got " + index);
        }
        return new DeviceSelector(backend, index, nameContains, capabilityRequirements);
    }

    /**
     * @param fragment matched case-insensitively against a device's {@link Device#displayName()}; a
     *     convenience for "the NVIDIA one", not an identity
     */
    public DeviceSelector withNameContaining(String fragment) {
        Objects.requireNonNull(fragment, "fragment");
        if (fragment.isBlank()) {
            throw new IllegalArgumentException("a name fragment must not be blank");
        }
        return new DeviceSelector(
                backend, index, fragment.toLowerCase(Locale.ROOT), capabilityRequirements);
    }

    /** Resolution must reject a device that does not support all of these. */
    public DeviceSelector requiring(DeviceCapability... required) {
        return requiring(Set.of(required));
    }

    public DeviceSelector requiring(Collection<DeviceCapability> required) {
        Objects.requireNonNull(required, "required");
        return new DeviceSelector(backend, index, nameContains, Set.copyOf(required));
    }

    public Optional<BackendId> backendId() {
        return Optional.ofNullable(backend);
    }

    public OptionalInt index() {
        return index == null ? OptionalInt.empty() : OptionalInt.of(index);
    }

    public Optional<String> nameContains() {
        return Optional.ofNullable(nameContains);
    }

    public Set<DeviceCapability> capabilityRequirements() {
        return capabilityRequirements;
    }

    /** Whether a resolved device satisfies everything this selector constrains. */
    public boolean matches(Device device) {
        Objects.requireNonNull(device, "device");
        if (backend != null && !backend.equals(device.backend())) {
            return false;
        }
        if (nameContains != null
                && !device.displayName().toLowerCase(Locale.ROOT).contains(nameContains)) {
            return false;
        }
        // The index is the backend's own ordinal and is not carried on the resolved device, so it
        // is a resolution input rather than something checkable here.
        return device.capabilities().supportsAll(capabilityRequirements);
    }

    @Override
    public boolean equals(Object other) {
        return other instanceof DeviceSelector that
                && Objects.equals(backend, that.backend)
                && Objects.equals(index, that.index)
                && Objects.equals(nameContains, that.nameContains)
                && capabilityRequirements.equals(that.capabilityRequirements);
    }

    @Override
    public int hashCode() {
        return Objects.hash(backend, index, nameContains, capabilityRequirements);
    }

    @Override
    public String toString() {
        StringBuilder out = new StringBuilder("device[");
        out.append(backend == null ? "any backend" : backend.toString());
        if (index != null) {
            out.append(", index ").append(index);
        }
        if (nameContains != null) {
            out.append(", name contains '").append(nameContains).append('\'');
        }
        if (!capabilityRequirements.isEmpty()) {
            out.append(", requiring ")
                    .append(
                            new java.util.TreeSet<>(
                                    capabilityRequirements.stream()
                                            .map(DeviceCapability::name)
                                            .toList()));
        }
        return out.append(']').toString();
    }
}
