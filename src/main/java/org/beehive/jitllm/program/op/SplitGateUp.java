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
package org.beehive.jitllm.program.op;

import java.util.List;
import java.util.Objects;
import org.beehive.jitllm.runtime.tensor.DataType;

/**
 * Separating a fused gate/up feed-forward projection into its two halves.
 *
 * <p>Phi3 projects the gate and the up branches with <b>one</b> matrix of width {@code 2 ×
 * hiddenDim} and reads the halves out of the single result. The projection is one {@code MatVec} —
 * one matrix, one multiply — and this is what happens to its output before {@link SwiGLU} combines
 * the two.
 *
 * <p>The twin of {@link SplitFusedQkv}, added by the same decision and for the same reason: it
 * transforms a <b>per-invocation activation, after the projection has run</b>, on every token,
 * where materialization is a loading and storage-representation concern fixed once. A backend may
 * fuse the projection, this split and the {@code SwiGLU} into one kernel — Phi3's GPU path already
 * does, as {@code splitGateUpAndSiLU} — and still be doing exactly what it should. What is not
 * optional is that the program says the split happens.
 *
 * @param fused the projection result holding the gate and up halves end to end
 * @param gate where the gate half is written
 * @param up where the up half is written
 * @param width elements in each half; the projection produces {@code 2 × width}
 * @param dataType the representation the copy executes at
 */
public record SplitGateUp(
        OperandRef fused, OperandRef gate, OperandRef up, int width, DataType dataType)
        implements Operation {

    public SplitGateUp {
        Objects.requireNonNull(fused, "fused");
        Objects.requireNonNull(gate, "gate");
        Objects.requireNonNull(up, "up");
        Objects.requireNonNull(dataType, "dataType");
        if (width < 1) {
            throw new IllegalArgumentException("width must be at least 1: " + width);
        }
    }

    /** The width of the fused projection this reads, which is what the projection must produce. */
    public int fusedWidth() {
        return 2 * width;
    }

    @Override
    public OperationKind kind() {
        return OperationKind.SPLIT_GATE_UP;
    }

    @Override
    public List<OperandRef> inputs() {
        return List.of(fused);
    }

    @Override
    public List<OperandRef> outputs() {
        return List.of(gate, up);
    }
}
