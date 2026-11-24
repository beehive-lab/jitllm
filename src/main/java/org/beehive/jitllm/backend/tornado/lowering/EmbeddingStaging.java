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
package org.beehive.jitllm.backend.tornado.lowering;

import java.lang.foreign.MemorySegment;
import org.beehive.jitllm.inference.state.State;
import org.beehive.jitllm.inference.weights.tornado.TornadoWeights;
import org.beehive.jitllm.model.Model;

/** Staging one token's embedding into the carrier a compiled program actually reads. */
public final class EmbeddingStaging {

    private EmbeddingStaging() {}

    /** Copies {@code token}'s embedding row into {@code workspace}'s staging carrier. */
    public static void stage(Model model, State state, int token) {
        final int dim = model.configuration().dim();
        final TornadoWeights weights = (TornadoWeights) model.weights();
        switch (weights.dataType()) {
            case F16 -> {
                MemorySegment table =
                        weights.getTokenEmbeddingTable().asHalfFloatArray().getSegment();
                int bytes = Short.BYTES;
                MemorySegment.copy(
                        table,
                        (long) token * dim * bytes,
                        state.workspace.embeddingX.getSegment(),
                        0,
                        (long) dim * bytes);
            }
            case Q8_0 -> {
                MemorySegment table = weights.getTokenEmbeddingTable().asByteArray().getSegment();
                int blockSize = 32;
                int blockBytes = 34; // 2 bytes scale + 32 bytes quants
                int blocksPerToken = (dim + blockSize - 1) / blockSize;
                long bytesPerToken = (long) blocksPerToken * blockBytes;
                MemorySegment.copy(
                        table,
                        (long) token * bytesPerToken,
                        state.workspace.embeddingX.getSegment(),
                        0,
                        bytesPerToken);
            }
            default ->
                    throw new IllegalArgumentException(
                            "Unsupported weight type: " + weights.dataType());
        }
    }
}
