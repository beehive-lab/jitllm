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

import java.util.List;
import org.beehive.jitllm.program.op.Operation;

/**
 * Consecutive program operations of one layer that a plan runs through implementations, for example
 * a layer's operations before attention: RMS norm, then the q, k and v projections.
 *
 * @param layer the transformer layer
 * @param name a short name for messages, for example {@code "before attention"}
 * @param tokens the tokens of one chunk
 * @param ops the operations, in the order they run
 */
public record Region(int layer, String name, int tokens, List<Operation> ops) {

    public Region {
        ops = List.copyOf(ops);
    }
}
