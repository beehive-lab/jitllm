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
package org.beehive.jitllm.tensor.standard;

import static org.junit.Assert.assertEquals;

import org.beehive.jitllm.runtime.tensor.DataType;
import org.junit.Test;

public class ArrayHalfFloatTensorTest {

    @Test
    public void storesHalfPrecisionAndReadsItBackAsFloat() {
        FloatTensor t = ArrayHalfFloatTensor.allocate(2, 4);
        assertEquals(8, t.size());
        assertEquals(DataType.F16, t.dataType());
        t.setFloat(0, 1.5f);
        t.setFloat(1, -2.25f);
        t.setFloat(2, 0.1f);
        assertEquals(1.5f, t.getFloat(0), 0f);
        assertEquals(-2.25f, t.getFloat(1), 0f);
        // Rounded to the nearest binary16, not kept at FP32.
        assertEquals(Float.float16ToFloat(Float.floatToFloat16(0.1f)), t.getFloat(2), 0f);
        assertEquals(0.1f, t.getFloat(2), 1e-4f);
    }

    @Test
    public void copiesAndDotsThroughTheFloatTensorContract() {
        FloatTensor half = ArrayHalfFloatTensor.allocate(4);
        FloatTensor full = ArrayFloatTensor.allocate(4);
        for (int i = 0; i < 4; i++) {
            full.setFloat(i, i + 0.5f);
        }
        full.copyTo(0, half, 0, 4);
        assertEquals(
                0.5f * 0.5f + 1.5f * 1.5f + 2.5f * 2.5f + 3.5f * 3.5f, full.dot(0, half, 0, 4), 0f);
    }
}
