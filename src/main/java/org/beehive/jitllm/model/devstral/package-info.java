/**
 * Devstral 2 model support (Mistral 3 architecture).
 *
 * <h2>Key architectural differences from standard Mistral</h2>
 *
 * <ul>
 *   <li><b>Independent head dimension:</b> {@code head_dim=128 != dim/num_heads=160}. Q projection
 *       outputs {@code qDim = num_heads * head_dim = 4096}, not {@code dim = 5120}. K/V projection
 *       outputs {@code kvDim = num_kv_heads * head_dim = 1024}, not {@code 1280}.
 *   <li><b>GGUF metadata prefix:</b> {@code mistral3.*} (not {@code llama.*}). Head dimension read
 *       from {@code mistral3.attention.key_length}.
 *   <li><b>Tekken tokenizer:</b> GPT-2-style BPE with explicit merges list and {@code
 *       BYTE_ENCODER}, replacing Mistral's SentencePiece score-based BPE. Identified by {@code
 *       tokenizer.ggml.pre=tekken}.
 *   <li><b>YaRN RoPE scaling:</b> Precomputed frequencies with {@code mscale = 1 + 0.1 *
 *       logMultiplier * ln(factor)}, required for correct positional encoding even at short
 *       contexts.
 *   <li><b>Non-square Q/K/V projections on GPU:</b> Dedicated kernels ({@code
 *       fusedQKVMatmulQ8NonSquare}) that separate {@code inputDim} (5120) from {@code qDim} (4096).
 * </ul>
 *
 * @see DevstralConfiguration
 * @see Devstral
 * @see <a href="https://huggingface.co/unsloth/Devstral-Small-2-24B-Instruct-2512-GGUF">Devstral
 *     Small 2 24B GGUF</a>
 */
package org.beehive.jllm.model.devstral;
