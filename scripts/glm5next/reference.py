"""PyTorch reference for GLM-5.3-Flash (glm5next) on a tiny configuration.

The script
  1. builds a tiny model with the same layer types as GLM-5.3-Flash (KDA, sparse
     MLA with the k-pool indexer, dense and MoE feed-forward, 4 mHC streams, one
     MTP layer) and random weights from a fixed seed;
  2. writes it as a Hugging Face checkpoint with the official layout: config.json,
     model.safetensors.index.json and two shards, with FP8 (E4M3, 128x128 block
     scales), BF16 and F32 tensors where the real checkpoint has them;
  3. runs a naive float64 forward pass over a fixed token sequence, reading the
     weights back from that checkpoint, and writes per-layer goldens to a GGUF
     file of F32 tensors;
  4. with --check-transformers, runs the same checkpoint through the
     transformers Glm5Next implementation and compares hidden states and logits.

The forward pass follows transformers' modeling_glm5_next.py; the MTP layer
follows vLLM's glm5next/common/mtp.py (transformers does not implement it).

    python -I reference.py --out /path/to/tiny [--check-transformers]
"""

import argparse
import json
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gguf_io  # noqa: E402

PREFIX = "model.language_model."

TINY_TEXT_CONFIG = {
    "vocab_size": 384,
    "hidden_size": 64,
    "intermediate_size": 256,
    "moe_intermediate_size": 32,
    "num_hidden_layers": 9,
    "layer_types": ["linear_attention"] * 3 + ["deepseek_sparse_attention"]
    + ["linear_attention"] * 3 + ["deepseek_sparse_attention"] + ["linear_attention"],
    "mlp_layer_types": ["dense"] * 3 + ["sparse"] * 6,
    "indexer_types": ["full"] * 9,
    "first_k_dense_replace": 3,
    "num_attention_heads": 2,
    "num_key_value_heads": 2,
    "q_lora_rank": 64,
    "kv_lora_rank": 32,
    "qk_nope_head_dim": 32,
    "qk_rope_head_dim": 0,
    "qk_head_dim": 32,
    "v_head_dim": 32,
    "head_dim": 0,
    "mla_use_nope": True,
    "n_routed_experts": 16,
    "num_experts_per_tok": 4,
    "n_shared_experts": 1,
    "n_group": 1,
    "topk_group": 1,
    "norm_topk_prob": True,
    "routed_scaling_factor": 2.5,
    "scoring_func": "sigmoid",
    "topk_method": "noaux_tc",
    "moe_router_dtype": "float32",
    "swiglu_limit": 10.0,
    "hidden_act": "silu",
    "rms_norm_eps": 1e-5,
    "linear_attn_config": {
        "num_heads": 2,
        "head_dim": 32,
        "short_conv_kernel_size": 4,
        "gate_lower_bound": -5.0,
        "kda_layers": [0, 1, 2, 4, 5, 6, 8],
        "full_attn_layers": [3, 7],
    },
    "index_n_heads": 8,
    "index_head_dim": 32,
    "index_topk": 8,
    "index_kpool": 4,
    "index_kpool_always_select_tail": True,
    "index_kpool_compress": True,
    "hc_mult": 4,
    "hc_eps": 1e-6,
    "hc_sinkhorn_iters": 20,
    "mhc": True,
    "num_nextn_predict_layers": 1,
    "max_position_embeddings": 4096,
    "tie_word_embeddings": False,
    "attention_bias": False,
    "model_type": "glm5_next_text",
    "eos_token_id": [1],
    "pad_token_id": 0,
}

HC_POST_MULT = 2.0

# Positions whose indexer selection had a tie at the budget boundary.
TIES = []
# Every width a multiple of 128 (and every reduction of 64), so the batched prefill's projections
# take the int8 tensor-core GEMM path; five layers: three KDA, one sparse attention, one KDA.
MEDIUM_OVERRIDES = {
    "hidden_size": 256,
    "intermediate_size": 512,
    "moe_intermediate_size": 128,
    "num_hidden_layers": 5,
    "layer_types": ["linear_attention"] * 3 + ["deepseek_sparse_attention", "linear_attention"],
    "mlp_layer_types": ["dense"] * 3 + ["sparse"] * 2,
    "indexer_types": ["full"] * 5,
    "q_lora_rank": 256,
    "kv_lora_rank": 128,
    "qk_nope_head_dim": 128,
    "qk_head_dim": 128,
    "v_head_dim": 128,
    "linear_attn_config": {
        "num_heads": 2,
        "head_dim": 128,
        "short_conv_kernel_size": 4,
        "gate_lower_bound": -5.0,
        "kda_layers": [0, 1, 2, 4],
        "full_attn_layers": [3],
    },
    "index_n_heads": 4,
    "index_head_dim": 128,
}

SEQ_TOKENS = 40


# ------------------------------------------------------------------ weights

def fp8_tensors(layer_mlp_sparse, is_dsa):
    """Names (relative to the layer) that the real checkpoint stores as FP8."""
    names = []
    if is_dsa:
        names += ["self_attn.q_a_proj", "self_attn.q_b_proj", "self_attn.kv_a_proj_with_mqa", "self_attn.o_proj"]
    if layer_mlp_sparse:
        names += ["mlp.shared_experts.gate_proj", "mlp.shared_experts.up_proj", "mlp.shared_experts.down_proj"]
    else:
        names += ["mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]
    return names


def build_weights(cfg, seed):
    """Returns {hf_name: (storage, float32 array)}; storage is F32, BF16 or F8_E4M3."""
    g = torch.Generator().manual_seed(seed)
    D = cfg["hidden_size"]
    la = cfg["linear_attn_config"]
    KH, KD = la["num_heads"], la["head_dim"]
    H = cfg["num_attention_heads"]
    NOPE, VD = cfg["qk_nope_head_dim"], cfg["v_head_dim"]
    QL, KVL = cfg["q_lora_rank"], cfg["kv_lora_rank"]
    IH, ID = cfg["index_n_heads"], cfg["index_head_dim"]
    POOL = cfg["index_kpool"]
    E, MI, DI = cfg["n_routed_experts"], cfg["moe_intermediate_size"], cfg["intermediate_size"]
    N = cfg["hc_mult"]
    V = cfg["vocab_size"]
    W = {}

    def randn(*shape, std=1.0):
        return (torch.randn(*shape, generator=g, dtype=torch.float64) * std).float().numpy()

    def linear(name, out_dim, in_dim, gain=1.0, storage="BF16"):
        W[name] = (storage, randn(out_dim, in_dim, std=gain / math.sqrt(in_dim)))

    def norm(name, n):
        W[name] = ("BF16", 1.0 + randn(n, std=0.1))

    W["model.language_model.embed_tokens.weight"] = ("BF16", randn(V, D))
    W["lm_head.weight"] = ("BF16", randn(V, D, std=1.0 / math.sqrt(D)))
    W[PREFIX + "norm.weight"] = ("BF16", 1.0 + randn(D, std=0.1))

    n_layers = cfg["num_hidden_layers"] + cfg["num_nextn_predict_layers"]
    for i in range(n_layers):
        mtp = i >= cfg["num_hidden_layers"]
        p = f"{PREFIX}layers.{i}."
        is_dsa = mtp or cfg["layer_types"][i] != "linear_attention"
        sparse = mtp or cfg["mlp_layer_types"][i] == "sparse"
        fp8 = set(fp8_tensors(sparse, is_dsa))

        def st(rel):
            return "F8_E4M3" if rel in fp8 else "BF16"

        norm(p + "input_layernorm.weight", D)
        norm(p + "post_attention_layernorm.weight", D)
        if not mtp:
            for hc in ("attn", "ffn"):
                W[p + f"hc_{hc}_fn"] = ("BF16", randn((2 + N) * N, N * D, std=1.0 / math.sqrt(N * D)))
                W[p + f"hc_{hc}_base"] = ("F32", randn((2 + N) * N, std=0.5))
                W[p + f"hc_{hc}_scale"] = ("F32", 1.0 + randn(3, std=0.2))
        else:
            W[p + "eh_proj.weight"] = ("BF16", randn(D, 2 * D, std=1.0 / math.sqrt(2 * D)))
            norm(p + "enorm.weight", D)
            norm(p + "hnorm.weight", D)
            norm(p + "shared_head.norm.weight", D)

        a = p + "self_attn."
        if not is_dsa:
            C = KH * KD
            for proj in ("q_proj", "k_proj", "v_proj"):
                linear(a + proj + ".weight", C, D)
                W[a + proj[0] + "_conv1d.weight"] = ("BF16", randn(C, 1, cfg["linear_attn_config"]["short_conv_kernel_size"], std=0.5))
            linear(a + "f_a_proj.weight", KD, D)
            linear(a + "f_b_proj.weight", C, KD)
            W[a + "dt_bias"] = ("F32", randn(C, std=0.5))
            W[a + "A_log"] = ("F32", randn(KH, std=0.5))
            linear(a + "b_proj.weight", KH, D)
            linear(a + "g_a_proj.weight", KD, D)
            linear(a + "g_b_proj.weight", C, KD)
            norm(a + "o_norm.weight", KD)
            linear(a + "o_proj.weight", D, C)
        else:
            linear(a + "q_a_proj.weight", QL, D, storage=st("self_attn.q_a_proj"))
            norm(a + "q_a_layernorm.weight", QL)
            linear(a + "q_b_proj.weight", H * NOPE, QL, storage=st("self_attn.q_b_proj"))
            linear(a + "kv_a_proj_with_mqa.weight", KVL, D, storage=st("self_attn.kv_a_proj_with_mqa"))
            norm(a + "kv_a_layernorm.weight", KVL)
            linear(a + "kv_b_proj.weight", H * (NOPE + VD), KVL)
            linear(a + "o_proj.weight", D, H * VD, storage=st("self_attn.o_proj"))
            x = a + "indexer."
            linear(x + "wq_b.weight", IH * ID, QL)
            linear(x + "wk.weight", ID, D)
            W[x + "k_norm.weight"] = ("BF16", 1.0 + randn(ID, std=0.1))
            W[x + "k_norm.bias"] = ("BF16", randn(ID, std=0.1))
            linear(x + "weights_proj.weight", IH, D)
            W[x + "index_kpool_compress_ape"] = ("BF16", randn(POOL, ID, std=0.5))
            linear(x + "index_kpool_compress_gate", ID, D)

        m = p + "mlp."
        if sparse:
            linear(m + "gate.weight", E, D)
            W[m + "gate.e_score_correction_bias"] = ("F32", randn(E, std=0.1))
            for e in range(E):
                q = f"{m}experts.{e}."
                linear(q + "gate_proj.weight", MI, D, gain=2.0, storage="F8_E4M3")
                linear(q + "up_proj.weight", MI, D, gain=2.0, storage="F8_E4M3")
                linear(q + "down_proj.weight", D, MI, storage="F8_E4M3")
            for proj, o, n_in in (("gate_proj", MI, D), ("up_proj", MI, D), ("down_proj", D, MI)):
                linear(m + f"shared_experts.{proj}.weight", o, n_in, gain=2.0 if proj != "down_proj" else 1.0,
                       storage=st(f"mlp.shared_experts.{proj}"))
        else:
            # Gain 4 pushes some gate/up values past the SwiGLU clamp of 10.
            linear(m + "gate_proj.weight", DI, D, gain=4.0, storage=st("mlp.gate_proj"))
            linear(m + "up_proj.weight", DI, D, gain=4.0, storage=st("mlp.up_proj"))
            linear(m + "down_proj.weight", D, DI, storage=st("mlp.down_proj"))
    return W


def to_storage(storage, arr):
    """Returns the raw safetensors arrays: the value plus its _scale_inv for FP8."""
    t = torch.from_numpy(np.ascontiguousarray(arr))
    if storage == "F32":
        return [("", "F32", t.float().numpy())]
    if storage == "BF16":
        return [("", "BF16", t.to(torch.bfloat16).view(torch.int16).numpy().view(np.uint16))]
    rows, cols = arr.shape
    br, bc = -(-rows // 128), -(-cols // 128)
    scale = np.zeros((br, bc), dtype=np.float32)
    q = np.zeros((rows, cols), dtype=np.uint8)
    for i in range(br):
        for j in range(bc):
            blk = arr[i * 128:(i + 1) * 128, j * 128:(j + 1) * 128]
            s = max(float(np.abs(blk).max()) / 448.0, 1e-12)
            scale[i, j] = s
            q[i * 128:(i + 1) * 128, j * 128:(j + 1) * 128] = gguf_io.f32_to_e4m3fn(blk / s)
    return [("", "F8_E4M3", q), ("_scale_inv", "F32", scale)]


def write_checkpoint(out_dir, cfg, weights):
    os.makedirs(out_dir, exist_ok=True)
    tensors = {}
    for name, (storage, arr) in weights.items():
        for suffix, dt, raw in to_storage(storage, arr):
            tensors[name + suffix] = (dt, raw)
    names = sorted(tensors)
    # Two shards, split by layer index, like the real checkpoint's per-layer shards.
    shards = {"model-00001-of-00002.safetensors": {}, "model-00002-of-00002.safetensors": {}}
    for n in names:
        layer = n.split("layers.")[1].split(".")[0] if "layers." in n else None
        first = layer is None or int(layer) < 4
        shards["model-00001-of-00002.safetensors" if first else "model-00002-of-00002.safetensors"][n] = tensors[n]
    weight_map = {}
    for shard, ts in shards.items():
        gguf_io.write_safetensors(os.path.join(out_dir, shard), ts)
        for n in ts:
            weight_map[n] = shard
    with open(os.path.join(out_dir, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {}, "weight_map": weight_map}, f, indent=1)
    full = {
        "architectures": ["Glm5NextForConditionalGeneration"],
        "model_type": "glm5_next",
        "text_config": cfg,
        "quantization_config": {"activation_scheme": "dynamic", "fmt": "e4m3", "quant_method": "fp8",
                                 "weight_block_size": [128, 128]},
        "tie_word_embeddings": False,
    }
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(full, f, indent=2)


# ------------------------------------------------------------------ forward

class Weights:
    def __init__(self, ckpt):
        self.ckpt = ckpt

    def __call__(self, name):
        return torch.from_numpy(self.ckpt.tensor_f32(name)).double()

    def has(self, name):
        return self.ckpt.has(name)


def rms_norm(x, w, eps):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w


def mlp(x, w, p, limit):
    gate = (x @ w(p + "gate_proj.weight").T).clamp(max=limit)
    up = (x @ w(p + "up_proj.weight").T).clamp(-limit, limit)
    return (torch.nn.functional.silu(gate) * up) @ w(p + "down_proj.weight").T


def moe(x, w, p, cfg):
    limit = cfg["swiglu_limit"]
    logits = x @ w(p + "gate.weight").T
    scores = torch.sigmoid(logits)
    choice = scores + w(p + "gate.e_score_correction_bias")
    top = torch.topk(choice, cfg["num_experts_per_tok"], dim=-1).indices
    weights = scores.gather(-1, top)
    weights = weights / (weights.sum(-1, keepdim=True) + 1e-20) * cfg["routed_scaling_factor"]
    out = torch.zeros_like(x)
    for t in range(x.shape[0]):
        for j in range(top.shape[1]):
            e = int(top[t, j])
            out[t] += weights[t, j] * mlp(x[t:t + 1], w, f"{p}experts.{e}.", limit)[0]
    return out + mlp(x, w, p + "shared_experts.", limit), top


def hyper_connection(streams, w, p, cfg):
    """streams [T, N, D] -> post [T, N], comb [T, N, N], collapsed [T, D]."""
    T, N, D = streams.shape
    eps, iters = cfg["hc_eps"], cfg["hc_sinkhorn_iters"]
    flat = streams.reshape(T, N * D)
    flat = flat * torch.rsqrt(flat.pow(2).mean(-1, keepdim=True) + cfg["rms_norm_eps"])
    mixes = flat @ w(p + "_fn").T
    base, scale = w(p + "_base"), w(p + "_scale")
    pre = torch.sigmoid(mixes[:, :N] * scale[0] + base[:N]) + eps
    post = HC_POST_MULT * torch.sigmoid(mixes[:, N:2 * N] * scale[1] + base[N:2 * N])
    comb = (mixes[:, 2 * N:] * scale[2] + base[2 * N:]).view(T, N, N)
    comb = torch.softmax(comb, dim=-1) + eps
    comb = comb / (comb.sum(-2, keepdim=True) + eps)
    for _ in range(iters - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + eps)
        comb = comb / (comb.sum(-2, keepdim=True) + eps)
    collapsed = (pre.unsqueeze(-1) * streams).sum(1)
    return post, comb, collapsed


def hc_merge(post, comb, block_out, residual):
    """out[n] = post[n] * block_out + sum_m comb[m, n] * residual[m]."""
    return post.unsqueeze(-1) * block_out.unsqueeze(1) + torch.einsum("tmn,tmd->tnd", comb, residual)


def l2norm(x, eps=1e-6):
    return x / torch.sqrt((x * x).sum(-1, keepdim=True) + eps)


def kda(x, w, p, cfg):
    """Kimi delta attention over x [T, D], token by token; returns output and final state."""
    la = cfg["linear_attn_config"]
    H, K = la["num_heads"], la["head_dim"]
    T = x.shape[0]
    width = la["short_conv_kernel_size"]
    eps = cfg["rms_norm_eps"]

    def conv_silu(proj, conv):
        y = x @ w(p + proj).T  # [T, C]
        cw = w(p + conv)[:, 0, :]  # [C, width]
        padded = torch.cat([torch.zeros(width - 1, y.shape[1], dtype=y.dtype), y], 0)
        out = torch.zeros_like(y)
        for t in range(T):
            out[t] = (padded[t:t + width].T * cw).sum(-1)
        return torch.nn.functional.silu(out)

    q = conv_silu("q_proj.weight", "q_conv1d.weight").view(T, H, K)
    k = conv_silu("k_proj.weight", "k_conv1d.weight").view(T, H, K)
    v = conv_silu("v_proj.weight", "v_conv1d.weight").view(T, H, K)
    graw = (x @ w(p + "f_a_proj.weight").T) @ w(p + "f_b_proj.weight").T + w(p + "dt_bias")
    decay_rate = torch.exp(w(p + "A_log")).view(1, H, 1)
    g = la["gate_lower_bound"] * torch.sigmoid(decay_rate * graw.view(T, H, K))  # log decay per channel
    beta = torch.sigmoid(x @ w(p + "b_proj.weight").T)  # [T, H]
    q = l2norm(q) / math.sqrt(K)
    k = l2norm(k)
    state = torch.zeros(H, K, K, dtype=x.dtype)  # [head, key, value]
    o = torch.zeros(T, H, K, dtype=x.dtype)
    for t in range(T):
        state = state * torch.exp(g[t]).unsqueeze(-1)
        kv_mem = (state * k[t].unsqueeze(-1)).sum(-2)
        delta = (v[t] - kv_mem) * beta[t].unsqueeze(-1)
        state = state + k[t].unsqueeze(-1) * delta.unsqueeze(-2)
        o[t] = (state * q[t].unsqueeze(-1)).sum(-2)
    gate = ((x @ w(p + "g_a_proj.weight").T) @ w(p + "g_b_proj.weight").T).view(T, H, K)
    o = rms_norm(o, w(p + "o_norm.weight"), eps) * torch.sigmoid(gate)
    return o.reshape(T, H * K) @ w(p + "o_proj.weight").T, state


def indexer_select(x, q_resid, w, p, cfg):
    """Returns, per query t, the sorted list of attended key positions."""
    IH, ID = cfg["index_n_heads"], cfg["index_head_dim"]
    pool = cfg["index_kpool"]
    T = x.shape[0]
    q = (q_resid @ w(p + "wq_b.weight").T).view(T, IH, ID)
    k = x @ w(p + "wk.weight").T
    k = torch.nn.functional.layer_norm(k, (ID,), w(p + "k_norm.weight"), w(p + "k_norm.bias"), eps=1e-6)
    gate = x @ w(p + "index_kpool_compress_gate").T  # [T, ID]
    ape = w(p + "index_kpool_compress_ape")  # [pool, ID]
    n_pools = T // pool  # complete pools in the cache
    pool_keys = torch.zeros(n_pools, ID, dtype=x.dtype)
    for j in range(n_pools):
        s = slice(j * pool, (j + 1) * pool)
        prob = torch.softmax(gate[s] + ape, dim=0)
        pool_keys[j] = (prob * k[s]).sum(0)
    head_w = (x @ w(p + "weights_proj.weight").T) * IH ** -0.5  # [T, IH]
    budget = cfg["index_topk"] // pool
    selected = []
    for t in range(T):
        visible = [j for j in range(n_pools) if j * pool + pool - 1 <= t]
        keys = []
        if visible:
            scores = torch.relu(q[t] @ pool_keys[visible].T * ID ** -0.5)  # [IH, P]
            index_scores = (head_w[t] @ scores).tolist()  # [P]
            # The best pools, ties to the lower pool. The official implementation leaves the order
            # of tied scores to torch.topk; a pool scores exactly 0 when every head's ReLU is 0.
            order = sorted(range(len(visible)), key=lambda o: (-index_scores[o], o))
            for o in order[:budget]:
                j = visible[o]
                keys += list(range(j * pool, (j + 1) * pool))
            boundary = sorted(index_scores, reverse=True)
            if len(boundary) > budget and boundary[budget - 1] == boundary[budget]:
                TIES.append(t)
        if cfg["index_kpool_always_select_tail"]:
            tail = (t + 1) % pool
            keys += list(range(t + 1 - tail, t + 1))
        selected.append(sorted(keys))
    return selected


def dsa(x, w, p, cfg):
    """Sparse MLA without RoPE over x [T, D]; returns output and the selected keys."""
    H, NOPE, VD = cfg["num_attention_heads"], cfg["qk_nope_head_dim"], cfg["v_head_dim"]
    eps = cfg["rms_norm_eps"]
    T = x.shape[0]
    q_resid = rms_norm(x @ w(p + "q_a_proj.weight").T, w(p + "q_a_layernorm.weight"), eps)
    q = (q_resid @ w(p + "q_b_proj.weight").T).view(T, H, NOPE)
    latent = rms_norm(x @ w(p + "kv_a_proj_with_mqa.weight").T, w(p + "kv_a_layernorm.weight"), eps)
    kv = (latent @ w(p + "kv_b_proj.weight").T).view(T, H, NOPE + VD)
    k, v = kv[..., :NOPE], kv[..., NOPE:]
    selected = indexer_select(x, q_resid, w, p + "indexer.", cfg)
    out = torch.zeros(T, H, VD, dtype=x.dtype)
    scale = NOPE ** -0.5
    for t in range(T):
        idx = torch.tensor(selected[t], dtype=torch.long)
        s = torch.einsum("hd,shd->hs", q[t], k[idx]) * scale
        out[t] = torch.einsum("hs,shd->hd", torch.softmax(s, -1), v[idx])
    return out.reshape(T, H * VD) @ w(p + "o_proj.weight").T, selected, latent


def forward(w, cfg, tokens):
    D, N = cfg["hidden_size"], cfg["hc_mult"]
    eps = cfg["rms_norm_eps"]
    T = len(tokens)
    gold = {}
    emb = w("model.language_model.embed_tokens.weight")[torch.tensor(tokens)]
    gold["embd"] = emb
    streams = emb.unsqueeze(1).expand(T, N, D).clone()
    for i in range(cfg["num_hidden_layers"]):
        p = f"{PREFIX}layers.{i}."
        residual = streams
        post, comb, x = hyper_connection(streams, w, p + "hc_attn", cfg)
        x = rms_norm(x, w(p + "input_layernorm.weight"), eps)
        gold[f"blk.{i}.attn_in"] = x
        if cfg["layer_types"][i] == "linear_attention":
            a, state = kda(x, w, p + "self_attn.", cfg)
            gold[f"blk.{i}.kda_state"] = state
        else:
            a, selected, latent = dsa(x, w, p + "self_attn.", cfg)
            gold[f"blk.{i}.topk"] = topk_tensor(selected, cfg)
            gold[f"blk.{i}.latent"] = latent
        gold[f"blk.{i}.attn_out"] = a
        streams = hc_merge(post, comb, a, residual)
        residual = streams
        post, comb, x = hyper_connection(streams, w, p + "hc_ffn", cfg)
        x = rms_norm(x, w(p + "post_attention_layernorm.weight"), eps)
        gold[f"blk.{i}.ffn_in"] = x
        if cfg["mlp_layer_types"][i] == "sparse":
            f, top = moe(x, w, p + "mlp.", cfg)
            gold[f"blk.{i}.experts"] = torch.sort(top, -1).values.double()
        else:
            f = mlp(x, w, p + "mlp.", cfg["swiglu_limit"])
        gold[f"blk.{i}.ffn_out"] = f
        streams = hc_merge(post, comb, f, residual)
        gold[f"blk.{i}.streams"] = streams
    hidden = rms_norm(streams.mean(1), w(PREFIX + "norm.weight"), eps)
    gold["hidden"] = hidden
    gold["logits"] = hidden @ w("lm_head.weight").T
    if cfg["num_nextn_predict_layers"]:
        mtp_hidden, mtp_logits = mtp_forward(w, cfg, tokens, hidden)
        gold["mtp.hidden"] = mtp_hidden
        gold["mtp.logits"] = mtp_logits
    return gold


def mtp_forward(w, cfg, tokens, hidden):
    """Next-next token prediction: position t sees token t+1 and the main model's hidden at t."""
    eps = cfg["rms_norm_eps"]
    i = cfg["num_hidden_layers"]
    p = f"{PREFIX}layers.{i}."
    T = len(tokens) - 1
    trunk_ties = len(TIES)  # transformers has no MTP layer to compare, so its ties do not matter
    emb = w("model.language_model.embed_tokens.weight")[torch.tensor(tokens[1:])]
    emb[0] = 0.0  # vLLM masks the embedding at position 0
    eh = torch.cat([rms_norm(emb, w(p + "enorm.weight"), eps),
                    rms_norm(hidden[:T], w(p + "hnorm.weight"), eps)], -1)
    h = eh @ w(p + "eh_proj.weight").T
    a, _, _ = dsa(rms_norm(h, w(p + "input_layernorm.weight"), eps), w, p + "self_attn.", cfg)
    residual = h + a
    f, _ = moe(rms_norm(residual, w(p + "post_attention_layernorm.weight"), eps), w, p + "mlp.", cfg)
    out = rms_norm(residual + f, w(p + "shared_head.norm.weight"), eps)
    del TIES[trunk_ties:]
    return out, out @ w("lm_head.weight").T


def topk_tensor(selected, cfg):
    width = cfg["index_topk"] + cfg["index_kpool"] - 1
    out = torch.full((len(selected), width), -1.0, dtype=torch.float64)
    for t, keys in enumerate(selected):
        out[t, :len(keys)] = torch.tensor(keys, dtype=torch.float64)
    return out


def write_goldens(path, gold, tokens, cfg):
    wr = gguf_io.GGUFWriter(path, "glm5next-goldens")
    wr.add_array("golden.tokens", gguf_io.GGUF_INT32, tokens)
    wr.add_uint32("golden.layers", cfg["num_hidden_layers"])
    arrays = {}
    for name, t in gold.items():
        a = t.detach().cpu().double().numpy().astype(np.float32)
        arrays[name] = a
        wr.add_tensor_info(name, list(reversed(a.shape)), gguf_io.F32)
    wr.write_header()
    for name, a in arrays.items():
        wr.write_tensor(name, a.tobytes())
    wr.close()


# ------------------------------------------------------------------ transformers cross-check

def check_transformers(ckpt_dir, cfg, tokens, gold):
    from transformers.models.glm5_next.configuration_glm5_next import Glm5NextTextConfig
    from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextModel

    text = dict(cfg)
    text["layer_types"] = ["indexed_attention" if t != "linear_attention" else t for t in cfg["layer_types"]]
    text["num_hidden_layers"] = cfg["num_hidden_layers"]
    text.pop("qk_head_dim", None)
    text.pop("model_type", None)
    tcfg = Glm5NextTextConfig(**{k: v for k, v in text.items()
                                if k not in ("mhc", "num_nextn_predict_layers", "first_k_dense_replace",
                                             "scoring_func", "topk_method", "moe_router_dtype",
                                             "mla_use_nope", "index_kpool_compress")})
    tcfg._attn_implementation = "eager"
    # The grouped-GEMM experts kernel has no float64 path.
    tcfg._experts_implementation = "eager"
    model = Glm5NextTextModel(tcfg).double().eval()
    ck = gguf_io.Checkpoint(ckpt_dir)
    sd = {}
    E = cfg["n_routed_experts"]
    for i in range(cfg["num_hidden_layers"]):
        p = f"{PREFIX}layers.{i}."
        for name in [n for n in ck.weight_map if n.startswith(p) and not n.endswith("_scale_inv")]:
            rel = name[len(PREFIX):]
            if ".experts." in rel or "_conv1d" in rel:
                continue
            rel = (rel.replace("self_attn.f_a_proj", "self_attn.forget_gate.f_a_proj")
                   .replace("self_attn.f_b_proj", "self_attn.forget_gate.f_b_proj")
                   .replace("self_attn.dt_bias", "self_attn.forget_gate.dt_bias")
                   .replace("self_attn.A_log", "self_attn.forget_gate.A_log")
                   .replace("hc_attn_fn", "attn_hc.fn").replace("hc_attn_base", "attn_hc.base")
                   .replace("hc_attn_scale", "attn_hc.scale").replace("hc_ffn_fn", "ffn_hc.fn")
                   .replace("hc_ffn_base", "ffn_hc.base").replace("hc_ffn_scale", "ffn_hc.scale"))
            sd[rel] = torch.from_numpy(ck.tensor_f32(name)).double()
        if cfg["layer_types"][i] == "linear_attention":
            sd[f"layers.{i}.self_attn.conv1d.weight"] = torch.cat(
                [torch.from_numpy(ck.tensor_f32(p + f"self_attn.{c}_conv1d.weight")).double() for c in "qkv"], 0)
        if cfg["mlp_layer_types"][i] == "sparse":
            gu, dn = [], []
            for e in range(E):
                q = f"{p}mlp.experts.{e}."
                gu.append(torch.cat([torch.from_numpy(ck.tensor_f32(q + "gate_proj.weight")),
                                     torch.from_numpy(ck.tensor_f32(q + "up_proj.weight"))], 0))
                dn.append(torch.from_numpy(ck.tensor_f32(q + "down_proj.weight")))
            sd[f"layers.{i}.mlp.experts.gate_up_proj"] = torch.stack(gu).double()
            sd[f"layers.{i}.mlp.experts.down_proj"] = torch.stack(dn).double()
    sd["embed_tokens.weight"] = torch.from_numpy(ck.tensor_f32(PREFIX + "embed_tokens.weight")).double()
    sd["norm.weight"] = torch.from_numpy(ck.tensor_f32(PREFIX + "norm.weight")).double()
    missing, unexpected = model.load_state_dict(sd, strict=False)
    missing = [m for m in missing if "e_score_correction_bias" not in m]
    if missing or unexpected:
        raise SystemExit(f"state dict mismatch: missing={missing} unexpected={unexpected}")
    for i in range(cfg["num_hidden_layers"]):
        if cfg["mlp_layer_types"][i] == "sparse":
            b = ck.tensor_f32(f"{PREFIX}layers.{i}.mlp.gate.e_score_correction_bias")
            model.layers[i].mlp.gate.e_score_correction_bias.copy_(torch.from_numpy(b).double())
    with torch.no_grad():
        out = model(input_ids=torch.tensor([tokens]), output_hidden_states=True, use_cache=False)
    worst = 0.0
    # hidden_states holds the embedding, every layer but the last, then the final normalized state.
    for i, hs in enumerate(out.hidden_states[1:-1]):
        ref = gold[f"blk.{i}.streams"]
        err = float((hs[0] - ref).abs().max() / ref.abs().max())
        worst = max(worst, err)
        print(f"layer {i}: max rel diff {err:.3e}")
    err = float((out.last_hidden_state[0] - gold["hidden"]).abs().max() / gold["hidden"].abs().max())
    print(f"final hidden: max rel diff {err:.3e}")
    worst = max(worst, err)
    lm = torch.from_numpy(ck.tensor_f32("lm_head.weight")).double()
    logits = out.last_hidden_state[0] @ lm.T
    err = float((logits - gold["logits"]).abs().max() / gold["logits"].abs().max())
    print(f"logits: max rel diff {err:.3e}")
    worst = max(worst, err)
    if TIES:
        raise SystemExit(f"indexer ties at positions {TIES}: the comparison is not meaningful")
    # transformers computes the mHC mixes and the router in float32.
    if worst > 1e-5:
        raise SystemExit(f"transformers disagrees with the reference: {worst:.3e}")
    print("transformers matches the reference")


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--seed", type=int, default=5301)
    ap.add_argument("--check-transformers", action="store_true")
    ap.add_argument("--config", choices=["tiny", "medium"], default="tiny",
                    help="tiny: the smallest model of every layer kind; medium: widths the int8 GEMMs take")
    args = ap.parse_args()

    cfg = json.loads(json.dumps(TINY_TEXT_CONFIG))
    if args.config == "medium":
        cfg.update(json.loads(json.dumps(MEDIUM_OVERRIDES)))
    ckpt_dir = os.path.join(args.out, "hf")
    write_checkpoint(ckpt_dir, cfg, build_weights(cfg, args.seed))

    rng = np.random.default_rng(args.seed)
    tokens = [int(t) for t in rng.integers(2, cfg["vocab_size"], SEQ_TOKENS)]
    w = Weights(gguf_io.Checkpoint(ckpt_dir))
    with torch.no_grad():
        gold = forward(w, cfg, tokens)
    write_goldens(os.path.join(args.out, "goldens.gguf"), gold, tokens, cfg)
    top = gold["logits"].argmax(-1).tolist()
    print(f"wrote {ckpt_dir} and goldens.gguf; tokens={tokens[:8]}...; argmax={top[:8]}...")
    if args.check_transformers:
        check_transformers(ckpt_dir, cfg, tokens, gold)


if __name__ == "__main__":
    main()
