"""PyTorch reference for DeepSeek-V4 (deepseek4) on a tiny configuration.

The script
  1. builds a tiny model with every layer kind of DeepSeek-V4 (hash and score routing,
     compression ratios 0, 4 with the indexer, and 128, the mHC streams and head, and the
     MTP layer) and random weights from a fixed seed;
  2. writes it as a checkpoint in the official layout (DeepSeek's tensor names, FP8 E4M3
     with 128x128 power-of-two block scales, BF16, F32 and I64 tensors where the real
     checkpoint has them; --expert-dtype fp4 writes the routed experts as packed FP4 with
     E8M0 scales, as V4-Flash does);
  3. runs DeepSeek's own inference/model.py over a fixed token sequence (official.py
     supplies PyTorch versions of its GPU kernels), three ways: the whole sequence as one
     prefill, one token at a time, and a prefill of the first tokens followed by single
     tokens; and writes the logits, each layer's streams, the indexer selections and the
     MTP logits to goldens.gguf, and the tokens to tokens.bin (int32).

The quantization-aware rounding of the reference is off unless --qat is given, as in
llama.cpp and transformers.

    python -I reference.py --out /path/to/tiny [--expert-dtype fp4] [--qat]

With --real, it instead writes goldens for a slice of layers of a real checkpoint (the shards
of those layers, the embedding and the head are enough), over a prompt read from --text:

    python -I reference.py --real /path/to/DeepSeek-V4-Flash-Base --layers 0-3 \
        --text prompt.txt --tokens 2304 --decode 64 --out /path/to/goldens
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "glm5next"))
import gguf_io  # noqa: E402
import official  # noqa: E402

TINY_CONFIG = {
    "architectures": ["DeepseekV4ForCausalLM"],
    "model_type": "deepseek_v4",
    "vocab_size": 320,
    "hidden_size": 128,
    "moe_intermediate_size": 32,
    "num_hidden_layers": 5,
    "num_hash_layers": 3,
    "num_nextn_predict_layers": 1,
    "num_attention_heads": 2,
    "num_key_value_heads": 1,
    "head_dim": 64,
    "qk_rope_head_dim": 32,
    "q_lora_rank": 64,
    "o_groups": 2,
    "o_lora_rank": 32,
    "sliding_window": 8,
    "compress_ratios": [0, 0, 4, 128, 4, 0],
    "compress_rope_theta": 160000,
    "rope_theta": 10000,
    "rope_scaling": {"type": "yarn", "factor": 16, "original_max_position_embeddings": 65536,
                     "beta_fast": 32, "beta_slow": 1},
    "max_position_embeddings": 1048576,
    "n_routed_experts": 8,
    "n_shared_experts": 1,
    "num_experts_per_tok": 2,
    "scoring_func": "sqrtsoftplus",
    "topk_method": "noaux_tc",
    "norm_topk_prob": True,
    "routed_scaling_factor": 1.5,
    "swiglu_limit": 10.0,
    "index_n_heads": 16,
    "index_head_dim": 64,
    "index_topk": 4,
    "hc_mult": 4,
    "hc_sinkhorn_iters": 20,
    "hc_eps": 1e-6,
    "rms_norm_eps": 1e-6,
    "tie_word_embeddings": False,
    "bos_token_id": 0,
    "eos_token_id": 1,
    "expert_dtype": "fp8",
    "quantization_config": {"activation_scheme": "dynamic", "fmt": "e4m3", "quant_method": "fp8",
                            "scale_fmt": "ue8m0", "weight_block_size": [128, 128]},
    "torch_dtype": "bfloat16",
}

# Widths the int8 GEMMs and the grouped expert GEMMs take: whole 128-column tiles.
MEDIUM_OVERRIDES = {
    "hidden_size": 256,
    "moe_intermediate_size": 128,
    "o_lora_rank": 128,
}


def model_args(cfg, max_seq_len):
    """The official ModelArgs for a Hugging Face config.json."""
    rs = cfg.get("rope_scaling") or {}
    return {
        "max_batch_size": 1,
        "max_seq_len": max_seq_len,
        "dtype": "fp8",
        "scale_fmt": "ue8m0",
        "expert_dtype": "fp4" if cfg.get("expert_dtype") == "fp4" else None,
        "scale_dtype": "fp32",
        "vocab_size": cfg["vocab_size"],
        "dim": cfg["hidden_size"],
        "moe_inter_dim": cfg["moe_intermediate_size"],
        "n_layers": cfg["num_hidden_layers"],
        "n_hash_layers": cfg["num_hash_layers"],
        "n_mtp_layers": cfg["num_nextn_predict_layers"],
        "n_heads": cfg["num_attention_heads"],
        "n_routed_experts": cfg["n_routed_experts"],
        "n_shared_experts": cfg["n_shared_experts"],
        "n_activated_experts": cfg["num_experts_per_tok"],
        "score_func": cfg["scoring_func"],
        "route_scale": cfg["routed_scaling_factor"],
        "swiglu_limit": cfg["swiglu_limit"],
        "q_lora_rank": cfg["q_lora_rank"],
        "head_dim": cfg["head_dim"],
        "rope_head_dim": cfg["qk_rope_head_dim"],
        "norm_eps": cfg["rms_norm_eps"],
        "o_groups": cfg["o_groups"],
        "o_lora_rank": cfg["o_lora_rank"],
        "window_size": cfg["sliding_window"],
        "compress_ratios": tuple(cfg["compress_ratios"]),
        "compress_rope_theta": cfg["compress_rope_theta"],
        "original_seq_len": rs.get("original_max_position_embeddings", 0),
        "rope_theta": cfg["rope_theta"],
        "rope_factor": rs.get("factor", 1),
        "beta_fast": rs.get("beta_fast", 32),
        "beta_slow": rs.get("beta_slow", 1),
        "index_n_heads": cfg["index_n_heads"],
        "index_head_dim": cfg["index_head_dim"],
        "index_topk": cfg["index_topk"],
        "hc_mult": cfg["hc_mult"],
        "hc_sinkhorn_iters": cfg["hc_sinkhorn_iters"],
        "hc_eps": cfg["hc_eps"],
    }


# ------------------------------------------------------------------ random checkpoint

def quantize_fp8_blocks(w, block=128):
    """FP8 E4M3 weight and F32 power-of-two scales per 128x128 block."""
    out_f, in_f = w.shape
    bo, bi = -(-out_f // block), -(-in_f // block)
    padded = torch.zeros(bo * block, bi * block)
    padded[:out_f, :in_f] = w
    blocks = padded.view(bo, block, bi, block)
    amax = blocks.abs().amax(dim=(1, 3))
    scale = torch.exp2(torch.ceil(torch.log2((amax / 448.0).clamp(min=1e-30))))
    q = (blocks / scale[:, None, :, None]).view(bo * block, bi * block)[:out_f, :in_f]
    return q.to(torch.float8_e4m3fn), scale.float()


def quantize_fp4_rows(w, block=32):
    """Packed FP4 E2M1 (two codes per int8, element 2i in the low nibble) and E8M0 scales."""
    out_f, in_f = w.shape
    b = w.view(out_f, in_f // block, block)
    scale = torch.exp2(torch.ceil(torch.log2((b.abs().amax(-1, keepdim=True) / 6.0).clamp(min=1e-30))))
    mag = (b.abs() / scale).unsqueeze(-1)
    code = (mag - official.FP4_VALUES[:8]).abs().argmin(-1) + 8 * (b < 0).long()
    code = code.view(out_f, in_f).to(torch.uint8)
    packed = (code[:, 0::2] | (code[:, 1::2] << 4)).view(torch.int8)
    return packed, scale.view(out_f, in_f // block).to(torch.float8_e8m0fnu)


def checkpoint_kind(name):
    if name.endswith((".ape", "attn_sink", "gate.bias")) or name.split(".")[-1].startswith("hc_"):
        return "F32"
    if name.endswith("tid2eid"):
        return "I64"
    if (name.endswith("norm.weight") or name in ("embed.weight", "head.weight")
            or name.endswith(("gate.weight", "weights_proj.weight", "compressor.wkv.weight",
                              "compressor.wgate.weight"))):
        return "BF16"
    return "QUANT"


def random_checkpoint(mod, cfg, seed):
    """Random weights for every parameter of the official model, in the checkpoint's dtypes."""
    g = torch.Generator().manual_seed(seed)
    model = official.build_empty(mod, model_args(cfg, 512))
    fp4 = cfg.get("expert_dtype") == "fp4"
    state = {}

    def randn(*shape, std=1.0):
        return torch.randn(*shape, generator=g) * std

    for name, p in model.named_parameters():
        if name.endswith(".scale") and not name.split(".")[-1].startswith("hc_"):
            continue  # written with its weight
        shape = tuple(p.shape)
        kind = checkpoint_kind(name)
        if kind == "F32":
            if name.endswith("_scale"):
                v = 0.5 + torch.rand(*shape, generator=g)
            elif name.endswith("_base") or name.endswith(".ape"):
                v = randn(*shape, std=0.5)
            elif name.endswith("_fn"):
                v = randn(*shape, std=0.5 / shape[1] ** 0.5)
            elif name.endswith("attn_sink"):
                v = randn(*shape)
            else:  # router bias
                v = randn(*shape, std=0.1)
            state[name] = v.float()
        elif kind == "I64":
            n_exp = cfg["n_routed_experts"]
            state[name] = torch.stack([torch.randperm(n_exp, generator=g)[:shape[1]]
                                       for _ in range(shape[0])]).long()
        elif kind == "BF16":
            if name.endswith("norm.weight"):
                v = 1 + randn(*shape, std=0.1)
            elif name == "embed.weight":
                v = randn(*shape)
            else:
                v = randn(*shape, std=1 / shape[-1] ** 0.5)
            state[name] = v.bfloat16()
        else:
            routed = ".experts." in name and ".shared_experts." not in name
            in_f = shape[1] * (2 if p.dtype == getattr(torch, "float4_e2m1fn_x2", None) else 1)
            out_f = shape[0]
            # large gate/up projections so the SwiGLU clamp takes effect for some values
            std = (4.0 if name.endswith(("w1.weight", "w3.weight")) else 1.0) / in_f ** 0.5
            w = randn(out_f, in_f, std=std)
            if routed and fp4:
                state[name], state[name.replace("weight", "scale")] = quantize_fp4_rows(w)
            else:
                state[name], state[name.replace("weight", "scale")] = quantize_fp8_blocks(w)
    for i in range(cfg["num_nextn_predict_layers"]):
        state[f"mtp.{i}.emb.tok_emb.weight"] = state["embed.weight"].clone()
        state[f"mtp.{i}.head.weight"] = state["head.weight"].clone()
    return state


def write_checkpoint(directory, cfg, state):
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    names = sorted(state)
    layers = [n for n in names if n.startswith("layers.")]
    shards = {"model-00001-of-00002.safetensors": [n for n in names if n not in layers],
              "model-00002-of-00002.safetensors": layers}
    weight_map = {}
    for shard, members in shards.items():
        save_file({n: state[n].contiguous() for n in members}, os.path.join(directory, shard))
        weight_map.update({n: shard for n in members})
    total = sum(t.numel() * t.element_size() for t in state.values())
    with open(os.path.join(directory, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {"total_size": total}, "weight_map": weight_map}, f, indent=1)


def read_checkpoint(directory):
    from safetensors.torch import load_file
    with open(os.path.join(directory, "model.safetensors.index.json")) as f:
        shards = sorted(set(json.load(f)["weight_map"].values()))
    state = {}
    for shard in shards:
        state.update(load_file(os.path.join(directory, shard)))
    return state


# ------------------------------------------------------------------ runs

def run(mod, margs, state, tokens, prefill, capture=False):
    """Runs the official model: a prefill of the first `prefill` tokens, then one token at a time.

    Returns the logits at every position and, with capture, each layer's streams and the indexer
    selections of the prefill.
    """
    n = len(tokens)
    margs = dict(margs, max_seq_len=max(512, -(-n // 128) * 128))
    model = official.build(mod, margs, state)
    ids = torch.tensor([tokens], dtype=torch.long)
    streams = {i: [] for i in range(len(model.layers))}
    selections = {}
    hooks = []
    if capture:
        for i, layer in enumerate(model.layers):
            hooks.append(layer.register_forward_hook(
                lambda m, a, out, i=i: streams[i].append(out.detach().clone())))
            idx = getattr(layer.attn, "indexer", None)
            if idx is not None:
                hooks.append(idx.register_forward_hook(
                    lambda m, a, out, i=i: selections.setdefault(i, []).append(out.detach().clone())))
    logits, finals = [], []
    with torch.inference_mode():
        pos = 0
        while pos < n:
            step = prefill if pos == 0 else 1
            chunk = ids[:, pos:pos + step]
            h = model.embed(chunk).unsqueeze(2).repeat(1, 1, model.hc_mult, 1)
            for layer in model.layers:
                h = layer(h, pos, chunk)
            finals.append(h)
            logits.append(official.all_logits(model, h)[0])
            pos += step
        h_all = torch.cat(finals, 1)
        mtp = None
        if model.mtp and capture:
            mtp = official.mtp_all_logits(model.mtp[0], h_all[:, :-1], 0, ids[:, 1:])[0]
    for hk in hooks:
        hk.remove()
    out = {"logits": torch.cat(logits, 0), "mtp_logits": mtp}
    if capture:
        out["streams"] = {i: torch.cat(v, 1)[0] for i, v in streams.items()}
        out["selections"] = {i: s[0][0].float() for i, s in selections.items()}
    del model
    return out


class GGUFWeights:
    """Official parameter names read from a deepseek4 GGUF written by convert.py, as float32."""

    def __init__(self, path):
        import convert
        self.g = gguf_io.GGUFReader(path)
        self.convert = convert
        self._cache = (None, None)

    def _gguf_name(self, name):
        for k, v in self.convert.TOP_NAMES.items():
            if name == k:
                return v, None
        _, i, rest = name.split(".", 2)
        if rest.startswith("ffn.experts."):
            e, proj, _ = rest[len("ffn.experts."):].split(".")
            g = dict(self.convert.EXPERT_PROJ)[proj]
            return f"blk.{i}.{g}", int(e)
        if rest == "ffn.gate.tid2eid":
            return f"blk.{i}.ffn_gate_tid2eid.weight", None
        return f"blk.{i}.{self.convert.LAYER_NAMES[rest]}", None

    def __contains__(self, name):
        try:
            return self._gguf_name(name)[0] in self.g.tensors
        except (KeyError, ValueError):
            return False

    def __getitem__(self, name):
        gname, expert = self._gguf_name(name)
        ne, t, off = self.g.tensors[gname]
        if t == 26:  # I32
            n = int(np.prod(ne))
            a = np.frombuffer(self.g._mm, np.int32, count=n, offset=self.g.data_start + off)
            return torch.from_numpy(a.reshape(list(reversed(ne))).copy())
        if expert is None:
            return torch.from_numpy(self.g.tensor_f32(gname).copy())
        # the parameters come expert by expert, so keep the layer's three projections decoded
        layer = gname.split(".")[1]
        if self._cache[0] != layer:
            self._cache = (layer, {})
        if gname not in self._cache[1]:
            self._cache[1][gname] = self.g.tensor_f32(gname)
        return torch.from_numpy(self._cache[1][gname][expert].copy())


def slice_args(cfg, first, last):
    """ModelArgs for decoder layers first..last of a checkpoint, renumbered from 0, without MTP."""
    m = model_args(cfg, 512)
    m["n_layers"] = last - first + 1
    m["n_hash_layers"] = max(0, min(cfg["num_hash_layers"], last + 1) - first)
    m["n_mtp_layers"] = 0
    m["compress_ratios"] = tuple(cfg["compress_ratios"][first:last + 1]) + (0,)
    return m


def run_real(args, mod):
    """Goldens for a slice of a real checkpoint: a prompt of --tokens tokens from --text, as one
    prefill and as a prefill followed by --decode single tokens."""
    from tokenizers import Tokenizer
    with open(os.path.join(args.real, "config.json")) as f:
        cfg = json.load(f)
    first, _, last = args.layers.partition("-")
    first, last = int(first), int(last or first)
    margs = slice_args(cfg, first, last)

    def rename(name):
        if name.startswith("layers."):
            i, rest = name[len("layers."):].split(".", 1)
            return f"layers.{first + int(i)}.{rest}"
        return name

    state = official.LazyCheckpoint(args.real, rename)
    if args.weights_gguf:
        # The weights as the GGUF stores them (Q8_0, BF16, ...), decoded to float32: comparing
        # with llama.cpp on that file then leaves only the arithmetic, not the weight rounding.
        margs = dict(margs, dtype="bf16", expert_dtype=None)
        state = GGUFWeights(args.weights_gguf)
    tok = Tokenizer.from_file(os.path.join(args.real, "tokenizer.json"))
    with open(args.text, encoding="utf-8") as f:
        ids = tok.encode(f.read()).ids
    tokens = ([cfg.get("bos_token_id", 0)] + ids)[:args.tokens]
    if len(tokens) < args.tokens:
        raise SystemExit(f"--text gives only {len(tokens)} tokens")

    full = run(mod, margs, state, tokens, prefill=len(tokens), capture=True)
    gold = {"logits": full["logits"]}
    for i, s in full["streams"].items():
        gold[f"layer{i}.streams"] = s
    for i, s in full["selections"].items():
        gold[f"layer{i}.indexer_selection"] = s
    del full
    if args.decode:
        mixed = run(mod, margs, state, tokens, prefill=len(tokens) - args.decode)
        d = (mixed["logits"] - gold["logits"]).abs().max().item()
        print(f"prefill {len(tokens) - args.decode} + {args.decode} single: max |logits - prefill logits| = {d:.3e}")
        gold["logits_mixed"] = mixed["logits"]
    os.makedirs(args.out, exist_ok=True)
    write_goldens(os.path.join(args.out, "goldens.gguf"), dict(num_hidden_layers=margs["n_layers"]), tokens, gold)
    np.asarray(tokens, dtype=np.int32).tofile(os.path.join(args.out, "tokens.bin"))
    print(f"wrote goldens.gguf and tokens.bin for layers {first}-{last}, {len(tokens)} tokens; "
          f"argmax={gold['logits'].argmax(-1)[:8].tolist()}...")


def write_goldens(path, cfg, tokens, gold):
    wr = gguf_io.GGUFWriter(path, "deepseek4-goldens")
    wr.add_array("golden.tokens", gguf_io.GGUF_INT32, tokens)
    wr.add_uint32("golden.layers", cfg["num_hidden_layers"])
    arrays = {k: v.detach().double().numpy().astype(np.float32) for k, v in gold.items()}
    for name, a in arrays.items():
        wr.add_tensor_info(name, list(reversed(a.shape)), gguf_io.F32)
    wr.write_header()
    for name, a in arrays.items():
        wr.write_tensor(name, a.tobytes())
    wr.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--official-dir", help="where model.py is (downloaded if missing; default <out>/official)")
    ap.add_argument("--seed", type=int, default=4001)
    ap.add_argument("--tokens", type=int, default=300)
    ap.add_argument("--prefill", type=int, default=67, help="prefill length of the mixed run")
    ap.add_argument("--expert-dtype", choices=["fp8", "fp4"], default="fp8")
    ap.add_argument("--qat", action="store_true", help="apply the reference's quantization-aware rounding")
    ap.add_argument("--config", choices=["tiny", "medium"], default="tiny",
                    help="tiny: the smallest model of every layer kind; medium: widths the grouped expert GEMMs take")
    ap.add_argument("--real", help="a real checkpoint directory: write goldens for a slice of it instead")
    ap.add_argument("--layers", default="0-3", help="with --real, the decoder layers (a-b)")
    ap.add_argument("--text", help="with --real, the prompt text")
    ap.add_argument("--decode", type=int, default=0, help="with --real, also decode the last N tokens one at a time")
    ap.add_argument("--weights-gguf", help="with --real, read the weights from this GGUF (convert.py's output) instead")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    model_py = official.fetch_model_py(args.official_dir or os.path.join(args.out, "official"))
    mod = official.load(model_py, qat=args.qat)
    official.math_check()
    if args.real:
        run_real(args, mod)
        return

    cfg = dict(TINY_CONFIG, expert_dtype=args.expert_dtype)
    if args.config == "medium":
        cfg.update(MEDIUM_OVERRIDES)
    ckpt_dir = os.path.join(args.out, "checkpoint")
    write_checkpoint(ckpt_dir, cfg, random_checkpoint(mod, cfg, args.seed))
    state = read_checkpoint(ckpt_dir)

    g = torch.Generator().manual_seed(args.seed + 1)
    tokens = torch.randint(0, cfg["vocab_size"], (args.tokens,), generator=g).tolist()

    margs = model_args(cfg, 512)
    full = run(mod, margs, state, tokens, prefill=len(tokens), capture=True)
    single = run(mod, margs, state, tokens, prefill=1)
    mixed = run(mod, margs, state, tokens, prefill=args.prefill)
    scale = full["logits"].abs().max().item()
    for name, other in (("one token at a time", single), (f"prefill {args.prefill} + single", mixed)):
        d = (other["logits"] - full["logits"]).abs().max().item()
        print(f"{name}: max |logits - prefill logits| = {d:.3e} (logits up to {scale:.2f})")

    gold = {"logits": full["logits"], "logits_single": single["logits"], "logits_mixed": mixed["logits"]}
    for i, s in full["streams"].items():
        gold[f"layer{i}.streams"] = s
    for i, s in full["selections"].items():
        gold[f"layer{i}.indexer_selection"] = s
    if full["mtp_logits"] is not None:
        gold["mtp_logits"] = full["mtp_logits"]
    write_goldens(os.path.join(args.out, "goldens.gguf"), cfg, tokens, gold)
    np.asarray(tokens, dtype=np.int32).tofile(os.path.join(args.out, "tokens.bin"))
    top = full["logits"].argmax(-1).tolist()
    print(f"wrote {ckpt_dir}, goldens.gguf and tokens.bin; tokens={tokens[:8]}...; argmax={top[:8]}...")


if __name__ == "__main__":
    main()
