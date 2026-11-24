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
package org.beehive.jitllm.inference.weights.standard;

import org.beehive.jitllm.runtime.tensor.DataType;
import org.beehive.jitllm.tensor.standard.FloatTensor;

public class Phi3StandardWeights extends StandardWeights {

    // Phi3-specific weight fields that don't exist in the base StandardWeights
    public final FloatTensor[] wqkv; // Combined query, key, value matrices (Phi3 format)
    public final FloatTensor[] wDown; // FFN down projection weight matrices
    public final FloatTensor[] wGateUp; // FFN gate and up projection weight matrices (combined)

    /**
     * Constructor for Phi3 standard (non-TornadoVM) mode Maps Phi3-specific weights to the standard
     * format and stores Phi3-specific weights
     *
     * @param token_embedding_table Token embeddings matrix
     * @param rms_att_weight RMSNorm weights for attention layers
     * @param wqkv Combined query, key, value weight matrices (Phi3 format)
     * @param wo Output projection matrices
     * @param rms_ffn_weight RMSNorm weights for FFN layers
     * @param wDown FFN down projection weight matrices
     * @param wGateUp FFN gate and up projection weight matrices (combined)
     * @param rms_final_weight Final layer normalization weights
     * @param freq_cis_real RoPE cosine components
     * @param freq_cis_imag RoPE sine components
     * @param wcls Classifier weights for output logits
     * @param weightType Weight type specification
     */
    public Phi3StandardWeights(
            FloatTensor token_embedding_table,
            FloatTensor[] rms_att_weight,
            FloatTensor[] wqkv,
            FloatTensor[] wo,
            FloatTensor[] rms_ffn_weight,
            FloatTensor[] wDown,
            FloatTensor[] wGateUp,
            FloatTensor rms_final_weight,
            FloatTensor freq_cis_real,
            FloatTensor freq_cis_imag,
            FloatTensor wcls,
            DataType weightType) {

        // Call parent constructor with standard format (using nulls for unsupported weights)
        super(
                token_embedding_table,
                rms_att_weight,
                null,
                null,
                null, // wq, wk, wv - not used in Phi3
                wo,
                rms_ffn_weight,
                null,
                null,
                null, // w1, w2, w3 - not used in Phi3
                rms_final_weight,
                freq_cis_real,
                freq_cis_imag,
                wcls,
                weightType);

        // Store Phi3-specific weights
        this.wqkv = wqkv;
        this.wDown = wDown;
        this.wGateUp = wGateUp;
    }

    @Override
    public DataType dataType() {
        return weightType;
    }
}
