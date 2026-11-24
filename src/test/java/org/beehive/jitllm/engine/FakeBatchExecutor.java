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
package org.beehive.jitllm.engine;

import java.util.ArrayList;
import java.util.List;
import org.beehive.jitllm.runtime.batch.BatchExecutor;
import org.beehive.jitllm.runtime.batch.BatchSlots;

/**
 * A backend that produces tokens without a device, so the engine's whole contract is testable.
 *
 * <p>Every slot emits {@code tokensBeforeStop} tokens and then a stop token. Records the width and
 * the active count of every step, so a test can assert that inactive slots were still passed
 * through — they are part of the batch, and an executor that saw them removed would hide the bug
 * where the engine stops sizing for B.
 */
final class FakeBatchExecutor implements BatchExecutor {

    static final int STOP_TOKEN = -99;

    private final int tokensBeforeStop;
    private final int maxBatchSize;

    final List<Integer> activeCountsPerStep = new ArrayList<>();
    final List<Integer> slotCountsPerStep = new ArrayList<>();

    FakeBatchExecutor(int maxBatchSize, int tokensBeforeStop) {
        this.maxBatchSize = maxBatchSize;
        this.tokensBeforeStop = tokensBeforeStop;
    }

    @Override
    public int maxBatchSize() {
        return maxBatchSize;
    }

    @Override
    public int[] decodeStep(BatchSlots batch) {
        int[] tokens = new int[batch.width()];
        for (int i = 0; i < batch.width(); i++) {
            if (!batch.active()[i]) {
                tokens[i] = Integer.MIN_VALUE; // ignored; an inactive slot's value is nothing
                continue;
            }
            // position is how many tokens this sequence has been fed, prompt included
            int fed = batch.positions()[i];
            tokens[i] = fed >= tokensBeforeStop ? STOP_TOKEN : 1000 + fed;
        }
        activeCountsPerStep.add(batch.activeCount());
        slotCountsPerStep.add(batch.width());
        return tokens;
    }

    @Override
    public boolean isStopToken(int token) {
        return token == STOP_TOKEN;
    }
}
