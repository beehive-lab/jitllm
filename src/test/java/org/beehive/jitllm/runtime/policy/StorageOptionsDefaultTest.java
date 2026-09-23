package org.beehive.jitllm.runtime.policy;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;

import org.beehive.jitllm.runtime.tensor.DataType;
import org.junit.After;
import org.junit.Test;

public class StorageOptionsDefaultTest {

    private final String fp32 = System.getProperty(StorageOptions.FP32_PROPERTY);
    private final String legacy = System.getProperty(StorageOptions.LEGACY_FP16_PROPERTY);

    @After
    public void restore() {
        set(StorageOptions.FP32_PROPERTY, fp32);
        set(StorageOptions.LEGACY_FP16_PROPERTY, legacy);
    }

    @Test
    public void fp16IsTheDefault() {
        set(StorageOptions.FP32_PROPERTY, null);
        set(StorageOptions.LEGACY_FP16_PROPERTY, null);
        assertEquals(DataType.F16, StorageOptions.fromSystemProperties().keyValueRepresentation());
    }

    @Test
    public void fp32IsAnExplicitChoice() {
        set(StorageOptions.LEGACY_FP16_PROPERTY, null);
        set(StorageOptions.FP32_PROPERTY, "true");
        assertEquals(DataType.F32, StorageOptions.fromSystemProperties().keyValueRepresentation());
        assertEquals(DataType.F32, StorageOptions.fp32().keyValueRepresentation());
        assertEquals(DataType.F16, StorageOptions.fp16().keyValueRepresentation());
    }

    @Test
    public void theLegacyPropertyIsHonouredWhenSet() {
        set(StorageOptions.FP32_PROPERTY, null);
        set(StorageOptions.LEGACY_FP16_PROPERTY, "false");
        assertEquals(DataType.F32, StorageOptions.fromSystemProperties().keyValueRepresentation());
        set(StorageOptions.LEGACY_FP16_PROPERTY, "true");
        assertEquals(DataType.F16, StorageOptions.fromSystemProperties().keyValueRepresentation());
    }

    @Test
    public void askingForBothIsRefused() {
        set(StorageOptions.FP32_PROPERTY, "true");
        set(StorageOptions.LEGACY_FP16_PROPERTY, "true");
        assertThrows(IllegalArgumentException.class, StorageOptions::fromSystemProperties);
    }

    private static void set(String key, String value) {
        if (value == null) System.clearProperty(key);
        else System.setProperty(key, value);
    }
}
