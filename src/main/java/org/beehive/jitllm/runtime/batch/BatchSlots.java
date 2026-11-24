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
package org.beehive.jitllm.runtime.batch;

/**
 * One step's worth of work, as positions in fixed-width arrays.
 *
 * <p>All three arrays are {@code maxBatchSize} long, always. An inactive slot is not absent — it is
 * present and marked inactive, because the kernels execute the whole batch every step and a slot
 * that vanished from the arrays would silently shift every slot after it.
 *
 * @param active per slot: does it hold a live sequence
 * @param tokens per slot: the token to feed it this step. Ignored where inactive
 * @param positions per slot: the sequence position this token occupies. Ignored where inactive
 * @param kvSlots per slot: its row in the KV block table. Ignored where inactive
 */
public record BatchSlots(boolean[] active, int[] tokens, int[] positions, int[] kvSlots) {

    public BatchSlots {
        if (active.length != tokens.length
                || active.length != positions.length
                || active.length != kvSlots.length) {
            throw new IllegalArgumentException(
                    "a batch is fixed width: got "
                            + active.length
                            + " active, "
                            + tokens.length
                            + " tokens, "
                            + positions.length
                            + " positions, "
                            + kvSlots.length
                            + " kv slots");
        }
    }

    public int width() {
        return active.length;
    }

    public int activeCount() {
        int n = 0;
        for (boolean a : active) {
            if (a) {
                n++;
            }
        }
        return n;
    }
}
