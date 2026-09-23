package org.beehive.jllm.runtime.policy;

import java.util.Objects;
import org.beehive.jllm.api.Experimental;
import org.beehive.jllm.runtime.tensor.DataType;

/**
 * How a model's key/value storage is shaped: the choices that are <b>not</b> execution policy.
 *
 * @param keyValueRepresentation how key/value entries are stored: {@link DataType#F32} or {@link
 *     DataType#F16}
 * @param sharedKeyValuePool whether standalone sessions address one shared pool rather than each
 *     allocating their own cache. Applies to standalone sessions only: engine-batched execution
 *     <i>is</i> the shared pool, so there is no engine setting to make
 */
@Experimental
public record StorageOptions(DataType keyValueRepresentation, boolean sharedKeyValuePool) {

    public StorageOptions {
        Objects.requireNonNull(keyValueRepresentation, "keyValueRepresentation");
        if (keyValueRepresentation != DataType.F32 && keyValueRepresentation != DataType.F16) {
            throw new IllegalArgumentException(
                    "key/value storage is F32 or F16, not " + keyValueRepresentation);
        }
    }

    /**
     * The name of the property that selects an FP32 key/value cache instead of the FP16 default.
     */
    public static final String FP32_PROPERTY = "jllm.kvcache.fp32";

    /**
     * The earlier spelling, which selected FP16 when FP32 was the default. Still honoured when set:
     * {@code true} is the default now, and {@code false} asks for FP32.
     */
    public static final String LEGACY_FP16_PROPERTY = "jllm.kvcache.fp16";

    /** Half-precision key/value storage, each session with its own cache — the default. */
    public static StorageOptions fp16() {
        return new StorageOptions(DataType.F16, false);
    }

    /** Single-precision key/value storage: the compatibility and numerical-reference choice. */
    public static StorageOptions fp32() {
        return new StorageOptions(DataType.F32, false);
    }

    /**
     * The defaults this build runs with, from the {@code llama.*} system properties.
     *
     * <p>FP16 unless {@value #FP32_PROPERTY} is {@code true}, or the legacy {@value
     * #LEGACY_FP16_PROPERTY} is {@code false}. Setting both to ask for different representations is
     * refused rather than resolved by precedence.
     *
     * <p>Read per call rather than folded into a constant, for the reason {@link
     * ExecutionPolicy#fromSystemProperties()} gives: a constant is the defect being removed.
     * Nothing calls this in a loop — a model resolves it once, at load.
     */
    public static StorageOptions fromSystemProperties() {
        boolean fp32 = Boolean.getBoolean(FP32_PROPERTY);
        String legacy = System.getProperty(LEGACY_FP16_PROPERTY);
        if (fp32 && Boolean.parseBoolean(legacy)) {
            throw new IllegalArgumentException(
                    "-D"
                            + FP32_PROPERTY
                            + "=true and -D"
                            + LEGACY_FP16_PROPERTY
                            + "=true ask for different key/value caches; FP16 is the default, so"
                            + " drop "
                            + LEGACY_FP16_PROPERTY
                            + " and keep "
                            + FP32_PROPERTY
                            + " only if you want FP32");
        }
        boolean useFp32 = fp32 || (legacy != null && !Boolean.parseBoolean(legacy));
        return new StorageOptions(
                useFp32 ? DataType.F32 : DataType.F16, Boolean.getBoolean("jllm.kv.sharedPool"));
    }

    /** Whether key/value entries are half precision. */
    public boolean usesFp16KeyValueCache() {
        return keyValueRepresentation == DataType.F16;
    }
}
