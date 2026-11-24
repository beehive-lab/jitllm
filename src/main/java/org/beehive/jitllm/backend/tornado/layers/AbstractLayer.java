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
package org.beehive.jitllm.backend.tornado.layers;

import org.beehive.jitllm.inference.state.State;
import org.beehive.jitllm.inference.weights.Weights;
import org.beehive.jitllm.model.Configuration;
import uk.ac.manchester.tornado.api.GridScheduler;
import uk.ac.manchester.tornado.api.KernelContext;
import uk.ac.manchester.tornado.api.TaskGraph;

/** Abstract base class for activation, transformer-layer, and logits TaskGraphs. */
public abstract class AbstractLayer {

    /** Common constants used in tasks & worker-grid sizing. */
    protected static final int LOCAL_WORK_GROUP_SIZE_ALLOC = 32;

    protected static final int THREAD_SCALE_FOR_LOGITS = 8;

    protected final Weights weights;
    protected final Configuration config;
    protected final State state;
    protected final KernelContext context = new KernelContext();

    protected AbstractLayer(State state, Weights weights, Configuration config) {
        this.state = state;
        this.weights = weights;
        this.config = config;
    }

    /** Ensures weights are of the expected type. */
    @SuppressWarnings("unchecked")
    protected static <T> T requireWeightsType(
            Object weights, Class<T> expectedType, String layerName, String layout) {
        if (expectedType.isInstance(weights)) {
            return (T) weights;
        }
        throw new IllegalArgumentException(
                layerName
                        + " requires "
                        + expectedType.getSimpleName()
                        + " with "
                        + layout
                        + " layout");
    }

    public abstract GridScheduler updateGridScheduler(GridScheduler scheduler);

    /** Allow subclasses to override if they need custom data transfers. */
    protected TaskGraph configureLayerDataTransfers(TaskGraph tg, int layerIndex) {
        return tg;
    }
}
