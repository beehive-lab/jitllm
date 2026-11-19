"""Runs DeepSeek's official DeepSeek-V4 inference code (inference/model.py) on the CPU.

model.py imports its GPU kernels from kernel.py (tilelang) and its Hadamard rotation from
fast_hadamard_transform. This module provides plain PyTorch versions of those functions, in
float32, and loads model.py against them:

  - act_quant / fp4_act_quant: the quantization-aware rounding the reference applies to
    activations. Off by default (llama.cpp and transformers do not round); qat=True rounds
    to FP8 E4M3 or FP4 E2M1 with power-of-two block scales.
  - fp8_gemm / fp4_gemm: dequantize the weight with its block scales and multiply in float32.
  - sparse_attn: attention over the gathered top-k positions with the per-head sink.
  - hc_split_sinkhorn: the mHC pre/post/comb split and Sinkhorn normalization.
"""

import importlib.util
import math
import os
import subprocess
import sys
import types

import torch

MODEL_PY_URL = "https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/resolve/main/inference/model.py"

FP4_VALUES = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                           -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], dtype=torch.float32)


def fetch_model_py(directory):
    path = os.path.join(directory, "model.py")
    if not os.path.exists(path):
        os.makedirs(directory, exist_ok=True)
        subprocess.run(["curl", "-fsSL", "-o", path, MODEL_PY_URL], check=True)
    return path


def _pow2_ceil(x):
    return torch.exp2(torch.ceil(torch.log2(x.clamp(min=torch.finfo(torch.float32).tiny))))


def fp8_round(x, block, pow2_scale):
    """Rounds x to FP8 E4M3 per block of `block` values along the last dimension and back."""
    shape = x.shape
    b = x.float().reshape(-1, shape[-1] // block, block)
    scale = b.abs().amax(-1, keepdim=True) / 448.0
    if pow2_scale:
        scale = _pow2_ceil(scale)
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    q = (b / scale).clamp(-448, 448).to(torch.float8_e4m3fn).float() * scale
    return q.reshape(shape).to(x.dtype)


def fp4_round(x, block=32):
    """Rounds x to FP4 E2M1 per block of `block` values with a power-of-two scale and back."""
    shape = x.shape
    b = x.float().reshape(-1, shape[-1] // block, block)
    scale = _pow2_ceil(b.abs().amax(-1, keepdim=True) / 6.0)
    grid = FP4_VALUES[:8]
    mag = (b.abs() / scale).unsqueeze(-1)
    q = grid[(mag - grid).abs().argmin(-1)] * torch.sign(b)
    return (q * scale).reshape(shape).to(x.dtype)


def dequant_fp8(w, scale, block=128):
    out_f, in_f = w.shape
    s = scale.float().repeat_interleave(block, 0)[:out_f].repeat_interleave(block, 1)[:, :in_f]
    return w.float() * s


def dequant_fp4(w, scale, block=32):
    u = w.view(torch.uint8)
    vals = torch.stack([FP4_VALUES[(u & 0x0F).long()], FP4_VALUES[(u >> 4).long()]], -1).flatten(-2)
    return vals * scale.float().repeat_interleave(block, 1)


def hadamard(x):
    n = x.size(-1)
    h = torch.ones(1, 1, dtype=torch.float32)
    while h.size(0) < n:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    if h.size(0) != n:
        raise ValueError(f"Hadamard rotation needs a power of two, got {n}")
    return (x.float() @ h * n ** -0.5).to(x.dtype)


def make_kernel_module(qat):
    k = types.ModuleType("kernel")

    def act_quant(x, block_size=128, scale_fmt=None, scale_dtype=torch.float32, inplace=False):
        if inplace:
            if qat:
                x.copy_(fp8_round(x, block_size, scale_fmt is not None))
            return x
        return (fp8_round(x, block_size, scale_fmt is not None) if qat else x), None

    def fp4_act_quant(x, block_size=32, inplace=False):
        if inplace:
            if qat:
                x.copy_(fp4_round(x, block_size))
            return x
        return (fp4_round(x, block_size) if qat else x), None

    def fp8_gemm(x, s, w, ws, scale_dtype=torch.float32):
        return (x.float() @ dequant_fp8(w, ws).t()).to(x.dtype)

    def fp4_gemm(x, s, w, ws, scale_dtype=torch.float32):
        return (x.float() @ dequant_fp4(w, ws).t()).to(x.dtype)

    def sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale):
        b, m, h, d = q.shape
        idx = topk_idxs.long()
        valid = idx >= 0
        gathered = torch.stack([kv[i, idx[i].clamp(min=0)] for i in range(b)])  # [b, m, k, d]
        scores = torch.einsum("bmhd,bmkd->bmhk", q.float(), gathered.float()) * softmax_scale
        scores = scores.masked_fill(~valid[:, :, None, :], float("-inf"))
        sink = attn_sink.float()[None, None, :]
        top = torch.maximum(scores.amax(-1), sink)
        e = torch.exp(scores - top[..., None])
        denom = e.sum(-1) + torch.exp(sink - top)
        o = torch.einsum("bmhk,bmkd->bmhd", e, gathered.float()) / denom[..., None]
        return o.to(q.dtype)

    def hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult=4, sinkhorn_iters=20, eps=1e-6):
        hc = hc_mult
        pre = torch.sigmoid(mixes[..., :hc] * hc_scale[0] + hc_base[:hc]) + eps
        post = 2 * torch.sigmoid(mixes[..., hc:2 * hc] * hc_scale[1] + hc_base[hc:2 * hc])
        comb = (mixes[..., 2 * hc:] * hc_scale[2] + hc_base[2 * hc:]).unflatten(-1, (hc, hc))
        comb = comb.softmax(-1) + eps
        comb = comb / (comb.sum(-2, keepdim=True) + eps)
        for _ in range(sinkhorn_iters - 1):
            comb = comb / (comb.sum(-1, keepdim=True) + eps)
            comb = comb / (comb.sum(-2, keepdim=True) + eps)
        return pre, post, comb

    for f in (act_quant, fp4_act_quant, fp8_gemm, fp4_gemm, sparse_attn, hc_split_sinkhorn):
        setattr(k, f.__name__, f)
    return k


def load(model_py, qat=False):
    """Imports model.py with the PyTorch kernels; returns the module."""
    previous = sys.modules.get("kernel")
    sys.modules["kernel"] = make_kernel_module(qat)
    try:
        spec = importlib.util.spec_from_file_location("deepseek_v4_official_model", model_py)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        if previous is None:
            del sys.modules["kernel"]
        else:
            sys.modules["kernel"] = previous
    mod.rotate_activation = hadamard
    return mod


def build_empty(mod, args_dict):
    """The official Transformer with uninitialized parameters, activations in float32."""
    torch.set_default_dtype(torch.float32)
    return mod.Transformer(mod.ModelArgs(**args_dict))


class LazyCheckpoint:
    """Reads tensors of a checkpoint directory on demand, renaming model names to checkpoint names."""

    def __init__(self, directory, rename=lambda n: n):
        import json
        from safetensors import safe_open
        with open(os.path.join(directory, "model.safetensors.index.json")) as f:
            self.weight_map = json.load(f)["weight_map"]
        self.directory = directory
        self.rename = rename
        self._open = {}
        self._safe_open = safe_open

    def __contains__(self, name):
        return self.rename(name) in self.weight_map

    def __getitem__(self, name):
        name = self.rename(name)
        shard = self.weight_map[name]
        if shard not in self._open:
            self._open[shard] = self._safe_open(os.path.join(self.directory, shard), framework="pt")
        return self._open[shard].get_tensor(name)


def build(mod, args_dict, state):
    """Builds the official Transformer in float32 and loads its parameters from `state` (a dict or a
    LazyCheckpoint, keyed by checkpoint names).

    The checkpoint's FP8 wo_a is dequantized, and its BF16 tensors widened, the way the official
    convert.py and the float32 parameters expect. The MTP layer's copies of the embedding and head
    are not read: the model shares the trunk's.
    """
    model = build_empty(mod, args_dict)
    for name, p in model.named_parameters():
        if p.dtype == torch.bfloat16:
            p.data = p.data.float()
    for name, b in model.named_buffers():
        if b.dtype == torch.bfloat16:
            b.data = b.data.float()
    for name, p in model.named_parameters():
        if name not in state:
            raise SystemExit(f"parameter not in the checkpoint: {name}")
        t = state[name]
        if name.endswith("wo_a.weight") and name[:-len("weight")] + "scale" in state:
            t = dequant_fp8(t, state[name[:-len("weight")] + "scale"])
        if p.dtype in (torch.float8_e4m3fn, torch.float8_e8m0fnu, torch.int32, torch.int64):
            p.data.copy_(t.to(p.dtype) if t.dtype != p.dtype else t)
        elif p.dtype == getattr(torch, "float4_e2m1fn_x2", None):
            p.data.copy_(t.view(p.dtype))
        else:
            p.data.copy_(t.float())
    model.eval()
    return model


def all_logits(model, h):
    """The head over every position of the streams h [b, s, hc, d]."""
    x = model.head.hc_head(h, model.hc_head_fn, model.hc_head_scale, model.hc_head_base)
    return torch.nn.functional.linear(model.norm(x).float(), model.head.weight)


def mtp_all_logits(mtp, h, start_pos, next_ids):
    """The MTP layer over every position: streams h [b, s, hc, d], the next tokens [b, s]."""
    e = mtp.enorm(mtp.embed(next_ids))
    x = mtp.e_proj(e).unsqueeze(2) + mtp.h_proj(mtp.hnorm(h))
    x = mod_block_forward(mtp, x, start_pos, next_ids)
    y = mtp.head.hc_head(x, mtp.hc_head_fn, mtp.hc_head_scale, mtp.hc_head_base)
    return torch.nn.functional.linear(mtp.norm(y).float(), mtp.head.weight)


def mod_block_forward(block, x, start_pos, ids):
    """Block.forward of the MTP block, bypassing MTPBlock.forward (which keeps only the last logits)."""
    return type(block).__mro__[1].forward(block, x, start_pos, ids)


def math_check():
    """Self-test of the stand-in kernels."""
    x = torch.randn(3, 64)
    assert torch.allclose(hadamard(hadamard(x)), x, atol=1e-5)
    w = torch.randn(256, 256).to(torch.float8_e4m3fn)
    s = torch.exp2(torch.randint(-4, 4, (2, 2)).float())
    assert dequant_fp8(w, s).shape == (256, 256)
    assert math.isclose(float(fp8_round(torch.tensor([[1.0] * 64]), 64, True)[0, 0]), 1.0)
