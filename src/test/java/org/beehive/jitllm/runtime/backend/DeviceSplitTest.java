package org.beehive.jitllm.runtime.backend;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;

import java.util.List;
import org.junit.Test;

public class DeviceSplitTest {

    private static DeviceSplit split(int devices, Double... shares) {
        return new DeviceSplit(
                java.util.stream.IntStream.range(0, devices).mapToObj(d -> "0:" + d).toList(),
                List.of(shares),
                DeviceSplit.Transport.HOST);
    }

    @Test
    public void equalSharesSplitEvenly() {
        assertArrayEquals(new int[] {0, 40, 80}, split(2).layerBounds(80, 1));
        assertArrayEquals(new int[] {0, 8, 16}, split(2).layerBounds(16, 1));
    }

    @Test
    public void proportionsFollowTheTensorSplit() {
        assertArrayEquals(new int[] {0, 41, 80}, split(2, 41.0, 39.0).layerBounds(80, 1));
        assertArrayEquals(new int[] {0, 20, 60, 80}, split(3, 1.0, 2.0, 1.0).layerBounds(80, 1));
    }

    @Test
    public void everyStageKeepsAtLeastOneLayer() {
        assertArrayEquals(new int[] {0, 1, 2, 3}, split(3, 100.0, 1.0, 1.0).layerBounds(3, 1));
        assertArrayEquals(new int[] {0, 1, 4}, split(2, 1.0, 100.0).layerBounds(4, 1));
    }

    @Test
    public void innerBoundariesMoveToTheAlignment() {
        assertArrayEquals(new int[] {0, 32, 64}, split(2, 33.0, 31.0).layerBounds(64, 4));
        assertArrayEquals(new int[] {0, 4, 8}, split(2, 1.0, 100.0).layerBounds(8, 4));
    }

    private static long[] bytes(int layers, long each) {
        long[] bytes = new long[layers];
        java.util.Arrays.fill(bytes, each);
        return bytes;
    }

    @Test
    public void layersOfOneSizeSplitEvenly() {
        assertArrayEquals(new int[] {0, 40, 80}, split(2).layerBounds(bytes(80, 7), 1, 0, 0));
        assertArrayEquals(
                new int[] {0, 20, 40, 60, 80}, split(4).layerBounds(bytes(80, 7), 1, 0, 0));
    }

    /**
     * GLM-5.3-Flash with Q4_0 experts, in hundredths of a GiB: three dense layers of 0.29, then 42
     * expert layers of 3.96, the embedding table and the output projection 0.63 each. An equal
     * split puts twelve expert layers (47.5) on the second device; balanced, no device holds more
     * than eleven.
     */
    @Test
    public void unequalLayersSplitByTheirBytes() {
        long[] glm = bytes(45, 396);
        glm[0] = glm[1] = glm[2] = 29;
        assertArrayEquals(new int[] {0, 11, 23, 34, 45}, split(4).layerBounds(45, 1));
        assertArrayEquals(new int[] {0, 13, 24, 35, 45}, split(4).layerBounds(glm, 1, 63, 63));
    }

    @Test
    public void theLastStageCarriesTheOutputProjection() {
        assertArrayEquals(new int[] {0, 2, 4}, split(2).layerBounds(bytes(4, 1), 1, 0, 0));
        assertArrayEquals(new int[] {0, 3, 4}, split(2).layerBounds(bytes(4, 1), 1, 0, 3));
        assertArrayEquals(new int[] {0, 1, 4}, split(2).layerBounds(bytes(4, 1), 1, 3, 0));
    }

    @Test
    public void aTensorSplitOverridesTheBytes() {
        assertArrayEquals(
                new int[] {0, 41, 80}, split(2, 41.0, 39.0).layerBounds(bytes(80, 7), 1, 0, 0));
    }

    @Test
    public void bytesBalancedBoundariesStayOnTheAlignment() {
        assertArrayEquals(
                new int[] {0, 2, 6}, split(2).layerBounds(new long[] {5, 5, 1, 1, 1, 1}, 2, 0, 0));
        assertArrayEquals(
                new int[] {0, 4, 8},
                split(2).layerBounds(new long[] {1, 1, 1, 1, 1, 1, 1, 10}, 4, 0, 0));
        assertThrows(
                IllegalArgumentException.class, () -> split(3).layerBounds(bytes(8, 1), 4, 0, 0));
    }

    @Test
    public void invalidSplitsAreRefused() {
        assertThrows(IllegalArgumentException.class, () -> split(2, 1.0, 1.0, 1.0));
        assertThrows(IllegalArgumentException.class, () -> split(2, 1.0, 0.0));
        assertThrows(IllegalArgumentException.class, () -> split(2).layerBounds(1, 1));
        assertThrows(
                IllegalArgumentException.class,
                () -> new DeviceSplit(List.of("0"), List.of(), DeviceSplit.Transport.HOST));
    }

    @Test
    public void thePropertiesAreReadInOnePlace() {
        String devices = System.getProperty(DeviceSplit.DEVICES_PROPERTY);
        String shares = System.getProperty(DeviceSplit.SHARES_PROPERTY);
        String transport = System.getProperty(DeviceSplit.TRANSPORT_PROPERTY);
        try {
            System.clearProperty(DeviceSplit.DEVICES_PROPERTY);
            assertEquals(1, DeviceSplit.requestedDeviceCount());
            System.setProperty(DeviceSplit.DEVICES_PROPERTY, "0:0, 0:1");
            System.setProperty(DeviceSplit.SHARES_PROPERTY, "41,39");
            System.setProperty(DeviceSplit.TRANSPORT_PROPERTY, "host");
            DeviceSplit split = DeviceSplit.fromSystemProperties().orElseThrow();
            assertEquals(List.of("0:0", "0:1"), split.devices());
            assertEquals(List.of(41.0, 39.0), split.shares());
            assertEquals(DeviceSplit.Transport.HOST, split.transport());
            System.setProperty(DeviceSplit.TRANSPORT_PROPERTY, "pcie");
            assertThrows(IllegalArgumentException.class, DeviceSplit::fromSystemProperties);
        } finally {
            restore(DeviceSplit.DEVICES_PROPERTY, devices);
            restore(DeviceSplit.SHARES_PROPERTY, shares);
            restore(DeviceSplit.TRANSPORT_PROPERTY, transport);
        }
    }

    private static void restore(String key, String value) {
        if (value == null) {
            System.clearProperty(key);
        } else {
            System.setProperty(key, value);
        }
    }
}
