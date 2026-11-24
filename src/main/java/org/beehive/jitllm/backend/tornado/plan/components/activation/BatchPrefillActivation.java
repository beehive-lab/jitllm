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
package org.beehive.jitllm.backend.tornado.plan.components.activation;

import org.beehive.jitllm.backend.tornado.kernels.TransformerBatchPrefillKernels;
import org.beehive.jitllm.backend.tornado.kernels.TransformerComputeKernels;
import org.beehive.jitllm.backend.tornado.layers.ActivationTaskGraph;
import org.beehive.jitllm.backend.tornado.layers.GraphNames;
import org.beehive.jitllm.backend.tornado.scheduling.WorkerGridFactory;
import org.beehive.jitllm.inference.state.State;
import org.beehive.jitllm.model.Configuration;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.ImmutableTaskGraph;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;
import uk.ac.manchester.tornado.api.enums.DataTransferMode;

/**
 * Batch-prefill activation graph ("prefillActivation").
 *
 * <ul>
 *   <li>FP16: uploads {@code embeddingXBatch} (EVERY_EXECUTION) and converts to FP32 {@code
 *       wrapXBatch}.
 *   <li>Q8_0: uploads FP32 {@code wrapXBatch} pre-filled by host (EVERY_EXECUTION) and runs a no-op
 *       {@code batchPassthrough} so TornadoVM has at least one task.
 * </ul>
 */
public class BatchPrefillActivation implements ActivationTaskGraph {

    private final ImmutableTaskGraph itg;
    private final boolean isQ8;
    private final int batchSize;
    private final int dim;

    public BatchPrefillActivation(State state, Configuration config, int batchSize, boolean isQ8) {
        this.isQ8 = isQ8;
        this.batchSize = batchSize;
        this.dim = config.dim();
        KernelContext ctx = new KernelContext();
        this.itg = buildGraph(ctx, state).snapshot();
    }

    private TaskGraph buildGraph(KernelContext ctx, State state) {
        if (isQ8) {
            return new TaskGraph(GraphNames.PREFILL_ACTIVATION)
                    .transferToDevice(DataTransferMode.EVERY_EXECUTION, state.workspace.wrapXBatch)
                    .task(
                            "batchPassthrough",
                            TransformerBatchPrefillKernels::batchPassthrough,
                            ctx,
                            state.workspace.wrapXBatch)
                    .persistOnDevice(state.workspace.wrapXBatch);
        }
        return new TaskGraph(GraphNames.PREFILL_ACTIVATION)
                .transferToDevice(DataTransferMode.FIRST_EXECUTION, ctx, state.workspace.wrapXBatch)
                .transferToDevice(DataTransferMode.EVERY_EXECUTION, state.workspace.embeddingXBatch)
                .task(
                        "updateX",
                        TransformerComputeKernels::convertFP16toFP32,
                        ctx,
                        state.workspace.embeddingXBatch,
                        state.workspace.wrapXBatch)
                .persistOnDevice(state.workspace.wrapXBatch);
    }

    @Override
    public ImmutableTaskGraph getImmutableTaskGraph() {
        return itg;
    }

    @Override
    public GridScheduler updateGridScheduler(GridScheduler scheduler) {
        if (isQ8) {
            scheduler.addWorkerGrid(
                    "prefillActivation.batchPassthrough", WorkerGridFactory.genericWorker(1, 1));
        } else {
            scheduler.addWorkerGrid(
                    "prefillActivation.updateX",
                    WorkerGridFactory.genericWorker(batchSize * dim, 128));
        }
        return scheduler;
    }
}
