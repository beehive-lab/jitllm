## TornadoVM Transformer Optimizations

### Core Numerical Optimizations
- **Quantized Weight Support**
  - Kernels that read FP16, Q8_0 and Q4_0 weights directly
  - K-quants (Q4_K, Q5_K, Q6_K) are converted to Q8_0 at load time unless the model family has its own kernel for them

### Memory and Caching Optimizations
- **Key-Value Cache**
  - Efficiently stores past key-values for autoregressive generation
  - Organized by layer, position, and dimension for fast access
  - FP16 by default, half the memory of FP32; `--fp32-kv-cache` keeps FP32
- **Scale Caching**
  - Avoids redundant decompression of quantized weights
  - Caches scale factors for efficient block processing
- **Optimized GPU Memory Transfers**
  - Minimizes host-device data movement
  - One-time transfer of static data (weights, caches)
  - Per-execution transfer of dynamic data (position, activations)
- **Device-to-Device Data Consumption**
  - Efficient data transfer between operations
  - Reduces PCI-E bandwidth bottlenecks

### Algorithmic Optimizations
- **Parallel Reduction RMS Normalization**
  - Implements two-phase reduction for efficient normalization
  - Work group optimization for parallel sums
- **Rotary Position Embeddings (RoPE)**
  - Optimized implementation for positional encoding
  - Efficient rotation of query and key vectors
- **Optimized Float16 Decoding**
  - Fast decoder for half-precision floating point format
  - Special case handling for better performance
- **Parallelized Attention**
  - Computes attention heads in parallel
  - Optimized softmax with max subtraction for numerical stability
  - Split-KV decode attention: long contexts are split across workgroups and merged
- **Fused Feed-Forward Networks**
  - Combines operations for SwiGLU variant used in Llama models
  - Optimized SiLU and GELU activation functions

### GPU Execution Optimizations
- **Layered Execution Planning**
  - Organizes computation as separate layer-based task graphs
  - Strategic scheduling of operations
- **Work Group Optimization**
  - Tailored worker grid configurations for different operations
  - Matches GPU hardware characteristics
- **Local Memory Optimization**
  - Strategic use of local/shared memory for reductions
  - Optimizes bandwidth-intensive operations
- **Batched Prefill**
  - The prompt is processed 512 tokens per step on GPU by default
  - Tensor-core (MMA) GEMMs on CUDA, SIMD-group matrix kernels on Apple Metal
- **Vendor Native Libraries**
  - cuBLAS and cuDNN on CUDA tensor-core devices, MLX on Apple silicon, where installed
  - `--no-native-libraries` keeps jitllm's own kernels
- **CUDA Graphs**
  - `--cuda-graphs` captures and replays the task graphs to cut launch overhead
