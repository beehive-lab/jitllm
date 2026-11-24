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
package org.beehive.jitllm.model.granite;

import org.beehive.jitllm.model.Configuration;

// @formatter:off
public record GraniteConfiguration(
        String quantization,
        int dim,
        int hiddenDim,
        int numberOfLayers,
        int numberOfHeads,
        int numberOfKeyValueHeads,
        int vocabularySize,
        int contextLength,
        float rmsNormEps,
        float ropeTheta,
        // Granite-specific scaling factors (µP parameterization)
        float embeddingMultiplier, // multiply embeddings after lookup
        float residualMultiplier, // multiply residual additions
        float attentionMultiplier, // replaces 1/sqrt(headDim)
        float logitsScaling, // DIVIDE logits by this value
        boolean tieWordEmbeddings // share input/output embeddings
        ) implements Configuration {

    @Override
    public String quantization() {
        return quantization;
    }

    @Override
    public int numberOfHeadsKey() {
        // Granite uses standard GQA, same as Llama
        return numberOfKeyValueHeads;
    }

    @Override
    public int contextLengthModel() {
        return contextLength;
    }

    /** Size of each attention head (derived from dim / numberOfHeads) */
    @Override
    public int headSize() {
        return dim / numberOfHeads;
    }

    /** Key/value dimension (derived from dim * numberOfKeyValueHeads / numberOfHeads) */
    @Override
    public int kvDim() {
        return dim * numberOfKeyValueHeads / numberOfHeads;
    }

    /** Multiplier for key/value sharing in grouped-query attention */
    @Override
    public int kvMul() {
        return numberOfHeads / numberOfKeyValueHeads;
    }

    /**
     * Creates a new Configuration with a different context length.
     *
     * @param newContextLength The new context length to use
     * @return A new Configuration instance with updated context length, or the current instance if
     *     newContextLength is negative
     */
    // @formatter:off
    public GraniteConfiguration withContextLength(int newContextLength) {
        if (newContextLength < 0) {
            return this; // no change
        }
        return new GraniteConfiguration(
                this.quantization,
                this.dim,
                this.hiddenDim,
                this.numberOfLayers,
                this.numberOfHeads,
                this.numberOfKeyValueHeads,
                this.vocabularySize,
                newContextLength,
                this.rmsNormEps,
                this.ropeTheta,
                this.embeddingMultiplier,
                this.residualMultiplier,
                this.attentionMultiplier,
                this.logitsScaling,
                this.tieWordEmbeddings);
    }

    // @formatter:on

    /** Accessor for embedding scale (alias for embeddingMultiplier) */
    public float embeddingScale() {
        return embeddingMultiplier;
    }

    /** Accessor for residual scale (alias for residualMultiplier) */
    public float residualScale() {
        return residualMultiplier;
    }

    /** Accessor for attention scale (alias for attentionMultiplier) */
    public float attentionScale() {
        return attentionMultiplier;
    }

    /** Accessor for logit scale (alias for logitsScaling) */
    public float logitScale() {
        return logitsScaling;
    }

    /**
     * Factory method to create GraniteConfiguration with default scaling values for Granite 3.x
     * models.
     */
    public static GraniteConfiguration createDefault(
            String quantization,
            int dim,
            int hiddenDim,
            int numberOfLayers,
            int numberOfHeads,
            int numberOfKeyValueHeads,
            int vocabularySize,
            int contextLength,
            float rmsNormEps,
            float ropeTheta) {
        return new GraniteConfiguration(
                quantization,
                dim,
                hiddenDim,
                numberOfLayers,
                numberOfHeads,
                numberOfKeyValueHeads,
                vocabularySize,
                contextLength,
                rmsNormEps,
                ropeTheta,
                12.0f, // embeddingMultiplier
                0.22f, // residualMultiplier
                0.0078125f, // attentionMultiplier (1/128)
                16.0f, // logitsScaling (divisor)
                true // tieWordEmbeddings
                );
    }
}
