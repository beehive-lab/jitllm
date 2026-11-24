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
package org.beehive.jitllm.inference.weights.tornado;

import org.beehive.jitllm.backend.tornado.tensor.TornadoTensor;
import org.beehive.jitllm.runtime.tensor.DataType;

public class Qwen3TornadoWeights extends TornadoWeights {
    // Qwen3-specific fields
    public final TornadoTensor[] rms_att_KNormLayered;
    public final TornadoTensor[] rms_att_QNormLayered;

    // @formatter:off
    public Qwen3TornadoWeights(
            TornadoTensor tokenEmbeddingTable,
            TornadoTensor[] rmsAttWeight,
            TornadoTensor[] wq,
            TornadoTensor[] wk,
            TornadoTensor[] wv,
            TornadoTensor[] wo,
            TornadoTensor[] rms_att_KNormLayered,
            TornadoTensor[] rms_att_QNormLayered,
            TornadoTensor[] rmsFFNWeight,
            TornadoTensor[] w1,
            TornadoTensor[] w2,
            TornadoTensor[] w3,
            TornadoTensor rmsFinalWeight,
            TornadoTensor freqCisReal,
            TornadoTensor freqCisImag,
            TornadoTensor wCls,
            DataType weightType) {
        super(
                tokenEmbeddingTable,
                rmsAttWeight,
                wq,
                wk,
                wv,
                wo,
                rmsFFNWeight,
                w1,
                w2,
                w3,
                rmsFinalWeight,
                freqCisReal,
                freqCisImag,
                wCls,
                weightType);
        this.rms_att_KNormLayered = rms_att_KNormLayered;
        this.rms_att_QNormLayered = rms_att_QNormLayered;
    }
    // @formatter:on

}
