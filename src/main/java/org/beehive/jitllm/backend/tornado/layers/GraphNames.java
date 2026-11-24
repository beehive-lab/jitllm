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

/**
 * The names of the task graphs a plan chains together. A graph consumes buffers from another by
 * name, and worker grids are keyed by {@code graphName.taskName}, so every builder spells a name
 * the same way through here.
 */
public final class GraphNames {

    private GraphNames() {}

    /** The single-token activation graph: the embedding into {@code wrapX}. */
    public static final String ACTIVATION_UPDATE = "activationUpdate";

    /** The batched prefill's activation graph. */
    public static final String PREFILL_ACTIVATION = "prefillActivation";

    /** The decode activation that follows a batched prefill and relays its KV cache. */
    public static final String DECODE_ACTIVATION = "decodeActivation";

    /** The vocabulary projection graph. */
    public static final String LOGITS = "logits";

    /** A single-token or decode layer graph. */
    public static String layer(int index) {
        return "layer_" + index;
    }

    /** A batched-prefill layer graph. */
    public static String batchPrefillLayer(int index) {
        return "batchPrefillLayer_" + index;
    }

    /** A batched-decode layer graph. */
    public static String batchDecodeLayer(int index) {
        return "batchDecodeLayer_" + index;
    }

    /** Qwen3.5's name for its batched-prefill layer graphs. */
    public static String batchLayer(int index) {
        return "batchLayer_" + index;
    }
}
