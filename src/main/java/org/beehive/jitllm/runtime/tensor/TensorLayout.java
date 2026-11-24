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
package org.beehive.jitllm.runtime.tensor;

import java.util.Objects;

/**
 * How a tensor's values are physically organized.
 *
 * <p>Separate from {@link DataType} on purpose: the data type says what representation an operation
 * is parameterized by, the layout says how the bytes are arranged to hold it. Folding the two
 * together would make {@code Q8_0} with one scale arrangement a different data type from {@code
 * Q8_0} with another, and a data type would stop being the thing operations agree on.
 *
 * <p>Carries only what is needed to size, validate and materialize storage — no strides, no
 * offsets, no device handles.
 */
public sealed interface TensorLayout {

    /** Bytes one block occupies; 0 when values are stored individually. */
    int bytesPerBlock();

    /** Values per block; 1 when values are stored individually. */
    int valuesPerBlock();

    /** Bytes needed for {@code elementCount} values in this layout. */
    long byteSize(long elementCount);

    /** Values stored one after another, no blocks and no scales. */
    record Dense(int bytesPerValue) implements TensorLayout {

        public Dense {
            if (bytesPerValue <= 0) {
                throw new IllegalArgumentException(
                        "bytesPerValue must be positive: " + bytesPerValue);
            }
        }

        @Override
        public int bytesPerBlock() {
            return 0;
        }

        @Override
        public int valuesPerBlock() {
            return 1;
        }

        @Override
        public long byteSize(long elementCount) {
            return elementCount * bytesPerValue;
        }
    }

    /**
     * Values in fixed-size blocks, each with its own scale — the shape every quantization in this
     * engine uses.
     *
     * @param valuesPerBlock how many values share one scale
     * @param bytesPerBlock the block's total size, scale included
     * @param scale how the block's scale is stored
     */
    record BlockQuantized(int valuesPerBlock, int bytesPerBlock, ScaleFormat scale)
            implements TensorLayout {

        public BlockQuantized {
            if (valuesPerBlock <= 0 || bytesPerBlock <= 0) {
                throw new IllegalArgumentException(
                        "a block holds a positive number of values in a positive number of bytes");
            }
            Objects.requireNonNull(scale, "scale");
        }

        @Override
        public long byteSize(long elementCount) {
            long blocks = (elementCount + valuesPerBlock - 1) / valuesPerBlock;
            return blocks * bytesPerBlock;
        }
    }

    /** How a block's scale is represented. */
    enum ScaleFormat {

        /** One 16-bit float per block — Q8_0 and Q4_0. */
        FP16,

        /**
         * A block's scale is itself quantized against a super-block scale, as the K-quants do.
         * Named rather than described: nothing materializes these, so the details would be
         * decoration.
         */
        HIERARCHICAL
    }
}
