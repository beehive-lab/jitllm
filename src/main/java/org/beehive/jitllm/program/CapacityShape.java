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
package org.beehive.jitllm.program;

/**
 * What a compiled program's device arrays were sized from.
 *
 * @param contextLength the longest sequence the key/value storage was sized for
 * @param numberOfLayers how many transformer layers the program describes
 * @param kvBlockSize elements per key/value block
 * @param kvBlockCount how many blocks the pool holds
 * @param maxPrefillBatch the largest prefill batch the staging buffers were sized for
 * @param maxBatch the largest number of concurrent sequences, {@code B}
 */
public record CapacityShape(
        int contextLength,
        int numberOfLayers,
        int kvBlockSize,
        int kvBlockCount,
        int maxPrefillBatch,
        int maxBatch) {

    public CapacityShape {
        requirePositive("contextLength", contextLength);
        requirePositive("numberOfLayers", numberOfLayers);
        requirePositive("kvBlockSize", kvBlockSize);
        requirePositive("kvBlockCount", kvBlockCount);
        requirePositive("maxPrefillBatch", maxPrefillBatch);
        requirePositive("maxBatch", maxBatch);
    }

    private static void requirePositive(String what, int value) {
        if (value <= 0) {
            throw new IllegalArgumentException(what + " must be positive: " + value);
        }
    }
}
