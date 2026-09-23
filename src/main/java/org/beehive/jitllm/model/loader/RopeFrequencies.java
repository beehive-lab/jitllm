package org.beehive.jllm.model.loader;

import org.beehive.jllm.auxiliary.Pair;

/**
 * Precomputed rotary position frequencies, built once per model at load.
 *
 * <p><b>Load-time work, not an operation.</b> These tables are computed from the model's
 * configuration before any forward pass runs and are then read as an ordinary weight, in the {@code
 * ROPE_FREQUENCIES} role. The rotation itself is {@code program.op.RoPE}; this is the table it
 * reads.
 */
public final class RopeFrequencies {
    public static Pair<float[], float[]> precomputeFreqsCis(
            int contextLength,
            int headSize,
            double theta,
            boolean ropeScaling,
            float scaleFactor,
            float loFreqFactor,
            float hiFreqFactor,
            float oldContextLength) {
        assert headSize % 2 == 0;
        float[] cr = new float[contextLength * (headSize / 2)];
        float[] ci = new float[contextLength * (headSize / 2)];
        int n = 0;
        for (int pos = 0; pos < contextLength; ++pos) {
            for (int i = 0; i < headSize; i += 2) {
                float freq = (float) (1.0 / Math.pow(theta, i / (double) headSize));
                if (ropeScaling) {
                    // Llama 3.1 scaling
                    float loFreqWavelen = oldContextLength / loFreqFactor;
                    float hiFreqWavelen = oldContextLength / hiFreqFactor;
                    float wavelen = (float) (2.0 * Math.PI / freq);
                    if (wavelen < hiFreqWavelen) {
                        freq = freq;
                    } else if (wavelen > loFreqWavelen) {
                        freq = freq / scaleFactor;
                    } else {
                        float smooth =
                                (oldContextLength / wavelen - loFreqFactor)
                                        / (hiFreqFactor - loFreqFactor);
                        freq = (1.0f - smooth) * freq / scaleFactor + smooth * freq;
                    }
                }
                float val = pos * freq;
                cr[n] = (float) Math.cos(val);
                ci[n] = (float) Math.sin(val);
                n++;
            }
        }
        assert contextLength * (headSize / 2) == n;
        return new Pair<>(cr, ci);
    }

    public static Pair<float[], float[]> precomputeFreqsCisYaRN(
            int contextLength,
            int headSize,
            double theta,
            float factor,
            float betaFast,
            float betaSlow,
            float logMultiplier,
            int originalContextLength) {
        assert headSize % 2 == 0;
        float[] cr = new float[contextLength * (headSize / 2)];
        float[] ci = new float[contextLength * (headSize / 2)];

        float freqScale = 1.0f / factor;

        // Compute correlation dimensions for ramp interpolation
        float corrDim0 = yarnCorrDim(headSize, originalContextLength, betaFast, (float) theta);
        float corrDim1 = yarnCorrDim(headSize, originalContextLength, betaSlow, (float) theta);

        // Compute mscale (attention scaling for extended context)
        // Formula: mscale = 0.1 * logMultiplier * log(factor) + 1.0
        float mscale =
                logMultiplier > 0
                        ? 1.0f + 0.1f * logMultiplier * (float) Math.log(1.0f / freqScale)
                        : 1.0f;

        int n = 0;
        for (int pos = 0; pos < contextLength; ++pos) {
            for (int i = 0; i < headSize; i += 2) {
                float freqExtrap = (float) (1.0 / Math.pow(theta, i / (double) headSize));
                float freqInterp = freqScale * freqExtrap;

                float rampMix = yarnRamp(corrDim0, corrDim1, i / 2);
                float freq = freqInterp * (1.0f - rampMix) + freqExtrap * rampMix;

                float val = pos * freq;
                cr[n] = (float) Math.cos(val) * mscale;
                ci[n] = (float) Math.sin(val) * mscale;
                n++;
            }
        }
        assert contextLength * (headSize / 2) == n;
        return new Pair<>(cr, ci);
    }

    private static float yarnCorrDim(int nDims, int nCtxOrig, float nRot, float base) {
        return nDims
                * (float) Math.log(nCtxOrig / (nRot * 2.0f * (float) Math.PI))
                / (2.0f * (float) Math.log(base));
    }

    private static float yarnRamp(float low, float high, int i0) {
        float y = (i0 - low) / Math.max(0.001f, high - low);
        return 1.0f - Math.min(1.0f, Math.max(0.0f, y));
    }
}
