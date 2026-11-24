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
package org.beehive.jitllm.backend.tornado.ops;

import java.util.ArrayList;
import java.util.List;
import org.beehive.jitllm.program.op.MatMul;
import org.beehive.jitllm.program.op.Operation;

/**
 * Operations of one layer that one implementation runs together, because one of its fusions matched
 * them.
 *
 * @param layer the transformer layer
 * @param tokens the tokens of one chunk
 * @param fusion the fusion that matched
 * @param ops the matched operations, in order
 */
public record Step(int layer, int tokens, Fusion fusion, List<Operation> ops) {

    public Step {
        ops = List.copyOf(ops);
    }

    /** The projections among the operations, in order. */
    public List<MatMul> matMuls() {
        List<MatMul> matMuls = new ArrayList<>();
        for (Operation op : ops) {
            if (op instanceof MatMul m) {
                matMuls.add(m);
            }
        }
        return matMuls;
    }

    /** The first operation of type {@code type}. */
    public <T extends Operation> T first(Class<T> type) {
        for (Operation op : ops) {
            if (type.isInstance(op)) {
                return type.cast(op);
            }
        }
        throw new IllegalArgumentException(fusion.name() + " has no " + type.getSimpleName());
    }

    /** The number of weight values the projections multiply, all together. */
    public long weights() {
        long weights = 0;
        for (MatMul m : matMuls()) {
            weights += (long) m.rows() * m.columns();
        }
        return weights;
    }

    /** The number of values in one output row of the projections, all together. */
    public int outputWidth() {
        int width = 0;
        for (MatMul m : matMuls()) {
            width += m.rows();
        }
        return width;
    }
}
