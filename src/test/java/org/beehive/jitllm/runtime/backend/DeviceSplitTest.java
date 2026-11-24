/*
 * Copyright (c) 2024-2026, APT Group, Department of Computer Science,
 * The University of Manchester.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */
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
