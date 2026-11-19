"""Converts a DeepSeek-V4 checkpoint to llama.cpp's GGUF format (architecture deepseek4).

The output has the tensor names, shapes, types and metadata keys that llama.cpp's converter
(conversion/deepseek.py, DeepseekV4Model) writes, so llama.cpp and jitllm load the same file.
Unlike llama.cpp's converter, this one

  - converts V4-Flash-Base, whose routed experts are FP8: they become --expert-type (Q8_0 by
    default; Q4_0 or MXFP4 round them to 4 bits). V4-Flash's FP4 experts are repacked into
    MXFP4 blocks without loss by default, as llama.cpp does;
  - converts a contiguous slice of the decoder layers (--layers a-b, renumbered from 0) and
    reads, or with --hf-repo downloads, only the shards that hold them, the embedding and the
    head. --delete-shards removes each downloaded layer shard once it has been converted.

FP8 matrices are dequantized with their block scales and written as --type (Q8_0 by
default); BF16 matrices stay BF16; F32 tensors and every vector stay F32; the hash routing
tables become I32. --mtp adds the MTP layer to the same file (llama.cpp finds it there).

    python -I convert.py --model /path/to/checkpoint --out model.gguf [--layers 0-3]
    python -I convert.py --hf-repo deepseek-ai/DeepSeek-V4-Flash-Base --download-dir dl \\
        --layers 0-3 --delete-shards --out dsv4-l0-3-Q8_0.gguf
"""

import argparse
import json
import os
import struct
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "glm5next"))
import gguf_io  # noqa: E402

ARCH = "deepseek4"
I32, MXFP4 = 26, 39
gguf_io.TYPE_NAMES.update({I32: "I32", MXFP4: "MXFP4"})
gguf_io.TYPE_BY_NAME.update({"MXFP4": MXFP4})
gguf_io.BLOCK.update({I32: (1, 4), MXFP4: (32, 17)})
gguf_io._ST_DTYPES.setdefault("F8_E8M0", (np.uint8, 1))

FP4_VALUES = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                       -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], dtype=np.float32)

TOP_NAMES = {
    "embed.weight": "token_embd.weight",
    "norm.weight": "output_norm.weight",
    "head.weight": "output.weight",
    "hc_head_fn": "output_hc_fn.weight",
    "hc_head_base": "output_hc_base.weight",
    "hc_head_scale": "output_hc_scale.weight",
}

LAYER_NAMES = {
    "hc_attn_fn": "hc_attn_fn.weight",
    "hc_attn_base": "hc_attn_base.weight",
    "hc_attn_scale": "hc_attn_scale.weight",
    "hc_ffn_fn": "hc_ffn_fn.weight",
    "hc_ffn_base": "hc_ffn_base.weight",
    "hc_ffn_scale": "hc_ffn_scale.weight",
    "attn_norm.weight": "attn_norm.weight",
    "ffn_norm.weight": "ffn_norm.weight",
    "attn.attn_sink": "attn_sinks.weight",
    "attn.wq_a.weight": "attn_q_a.weight",
    "attn.q_norm.weight": "attn_q_a_norm.weight",
    "attn.wq_b.weight": "attn_q_b.weight",
    "attn.wkv.weight": "attn_kv.weight",
    "attn.kv_norm.weight": "attn_kv_a_norm.weight",
    "attn.wo_a.weight": "attn_output_a.weight",
    "attn.wo_b.weight": "attn_output_b.weight",
    "attn.compressor.ape": "attn_compressor_ape.weight",
    "attn.compressor.wkv.weight": "attn_compressor_kv.weight",
    "attn.compressor.wgate.weight": "attn_compressor_gate.weight",
    "attn.compressor.norm.weight": "attn_compressor_norm.weight",
    "attn.indexer.wq_b.weight": "indexer.attn_q_b.weight",
    "attn.indexer.weights_proj.weight": "indexer.proj.weight",
    "attn.indexer.compressor.ape": "indexer_compressor_ape.weight",
    "attn.indexer.compressor.wkv.weight": "indexer_compressor_kv.weight",
    "attn.indexer.compressor.wgate.weight": "indexer_compressor_gate.weight",
    "attn.indexer.compressor.norm.weight": "indexer_compressor_norm.weight",
    "ffn.gate.weight": "ffn_gate_inp.weight",
    "ffn.gate.bias": "exp_probs_b.bias",
    "ffn.shared_experts.w1.weight": "ffn_gate_shexp.weight",
    "ffn.shared_experts.w2.weight": "ffn_down_shexp.weight",
    "ffn.shared_experts.w3.weight": "ffn_up_shexp.weight",
}

MTP_NAMES = {
    "enorm.weight": "nextn.enorm.weight",
    "hnorm.weight": "nextn.hnorm.weight",
    "norm.weight": "nextn.shared_head_norm.weight",
}

EXPERT_PROJ = (("w1", "ffn_gate_exps.weight"), ("w2", "ffn_down_exps.weight"), ("w3", "ffn_up_exps.weight"))


def download(repo, directory, files):
    os.makedirs(directory, exist_ok=True)
    for name in files:
        target = os.path.join(directory, name)
        if os.path.exists(target) and not os.path.exists(target + ".part"):
            continue
        url = f"https://huggingface.co/{repo}/resolve/main/{name}"
        print(f"downloading {name}", flush=True)
        open(target + ".part", "a").close()
        subprocess.run(["curl", "-fL", "--retry", "5", "-C", "-", "-o", target, url], check=True)
        os.remove(target + ".part")


class Source:
    """The checkpoint's tensors as float32 (FP8 dequantized with its .scale), or raw.

    Shapes and types of a shard not downloaded yet come from its safetensors header, fetched from
    `repo` with a range request, so the whole file can be planned before the layers arrive.
    """

    def __init__(self, directory, repo=None):
        self.ckpt = gguf_io.Checkpoint(directory)
        self.directory = directory
        self.repo = repo
        self._headers = {}

    def has(self, name):
        return self.ckpt.has(name)

    def _header(self, name):
        shard = self.ckpt.shard_of(name)
        if os.path.exists(os.path.join(self.directory, shard)) or self.repo is None:
            return self.ckpt._file(name).header[name]
        if shard not in self._headers:
            url = f"https://huggingface.co/{self.repo}/resolve/main/{shard}"
            n = struct.unpack("<Q", subprocess.run(["curl", "-fsL", "-r", "0-7", url], check=True,
                                                   capture_output=True).stdout)[0]
            self._headers[shard] = json.loads(subprocess.run(["curl", "-fsL", "-r", f"8-{7 + n}", url],
                                                             check=True, capture_output=True).stdout)
        return self._headers[shard][name]

    def shape(self, name):
        return list(self._header(name)["shape"])

    def dtype(self, name):
        return self._header(name)["dtype"]

    def raw(self, name):
        return self.ckpt._file(name).raw(name)

    def f32(self, name):
        dt = self.dtype(name)
        if dt == "F8_E4M3":
            w = gguf_io.E4M3FN[self.raw(name)]
            return gguf_io.dequant_block_scale(w, self.f32(name[:-len("weight")] + "scale"))
        if dt == "F8_E8M0":
            return np.exp2(self.raw(name).astype(np.float32) - 127.0)
        if dt == "I8":  # packed FP4 E2M1 with a per-32 E8M0 scale
            codes = fp4_codes(self.raw(name))
            scale = self.f32(name[:-len("weight")] + "scale")
            return FP4_VALUES[codes] * np.repeat(scale, 32, axis=1)
        return self.ckpt.tensor_f32(name)


def fp4_codes(packed):
    u = packed.view(np.uint8)
    return np.stack([u & 0x0F, u >> 4], axis=-1).reshape(u.shape[0], -1)


def mxfp4_from_fp4(packed, scale_e8m0):
    """Repacks FP4 codes (element 2i in the low nibble) and E8M0 scales into ggml block_mxfp4:
    one scale byte, then 16 bytes holding element j in the low nibble and j+16 in the high one."""
    codes = fp4_codes(packed)
    rows, cols = codes.shape
    blocks = codes.reshape(rows, cols // 32, 32)
    qs = (blocks[..., :16] | (blocks[..., 16:] << 4)).astype(np.uint8)
    e = scale_e8m0.view(np.uint8).reshape(rows, cols // 32, 1)
    return np.concatenate([e, qs], axis=-1).tobytes()


def mxfp4_quantize(x):
    """Rounds float32 rows to MXFP4: a power-of-two scale per 32 values, nearest E2M1 code."""
    rows, cols = x.shape
    b = x.reshape(rows, cols // 32, 32)
    amax = np.abs(b).max(-1, keepdims=True)
    exp = np.ceil(np.log2(np.maximum(amax / 6.0, 2.0 ** -127)))
    mag = np.abs(b) / np.exp2(exp)
    code = np.abs(mag[..., None] - FP4_VALUES[:8]).argmin(-1) + 8 * (b < 0)
    qs = (code[..., :16] | (code[..., 16:] << 4)).astype(np.uint8)
    e = (exp + 127).clip(0, 254).astype(np.uint8)
    return np.concatenate([e, qs], axis=-1).tobytes()


class Plan:
    """The tensors to write: (gguf name, ne, type, producer of byte pieces)."""

    def __init__(self, src, target, small_type):
        self.src = src
        self.target = target
        self.small_type = small_type
        self.items = []

    def type_for(self, src_name, ne):
        dt = self.src.dtype(src_name)
        if len(ne) == 1 or dt == "F32":
            return gguf_io.F32
        if dt == "BF16":
            return self.small_type
        if ne[0] % 32:
            return gguf_io.F32
        return self.target

    def add(self, gguf_name, src_name, force_type=None):
        shape = self.src.shape(src_name)
        ne = list(reversed(shape))
        t = force_type if force_type is not None else self.type_for(src_name, ne)
        self.items.append((gguf_name, ne, t, lambda: [gguf_io.encode(self.src.f32(src_name), t)]))

    def add_i32(self, gguf_name, src_name):
        shape = self.src.shape(src_name)
        self.items.append((gguf_name, list(reversed(shape)), I32,
                           lambda: [np.ascontiguousarray(self.src.raw(src_name), dtype=np.int32).tobytes()]))

    def add_eh_proj(self, gguf_name, e_name, h_name):
        rows, cols = self.src.shape(e_name)
        ne = [2 * cols, rows]
        t = self.target if self.target != MXFP4 else gguf_io.Q8_0
        self.items.append((gguf_name, ne, t, lambda: [gguf_io.encode(
            np.concatenate([self.src.f32(e_name), self.src.f32(h_name)], axis=1), t)]))

    def add_experts(self, gguf_prefix, src_prefix, n_experts, expert_type):
        for proj, gname in EXPERT_PROJ:
            first = f"{src_prefix}{0}.{proj}.weight"
            fp4 = self.src.dtype(first) == "I8"
            rows, cols = self.src.shape(first)
            cols *= 2 if fp4 else 1
            t = expert_type if expert_type is not None else (MXFP4 if fp4 else self.target)

            def produce(proj=proj, fp4=fp4, t=t):
                for e in range(n_experts):
                    w = f"{src_prefix}{e}.{proj}.weight"
                    if t == MXFP4 and fp4:
                        yield mxfp4_from_fp4(self.src.raw(w), self.src.raw(w[:-len("weight")] + "scale"))
                    elif t == MXFP4:
                        yield mxfp4_quantize(self.src.f32(w))
                    else:
                        yield gguf_io.encode(self.src.f32(w), t)

            self.items.append((gguf_prefix + gname, [cols, rows, n_experts], t, produce))


def tokenizer_metadata(writer, model_dir, cfg):
    vocab_size = cfg["vocab_size"]
    tokens = [None] * vocab_size
    types = [1] * vocab_size
    path = os.path.join(model_dir, "tokenizer.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            tok = json.load(f)
        for text, i in tok["model"]["vocab"].items():
            tokens[i] = text
        merges = [m if isinstance(m, str) else " ".join(m) for m in tok["model"]["merges"]]
        for added in tok["added_tokens"]:
            tokens[added["id"]] = added["content"]
            types[added["id"]] = 3 if added["special"] else 4
    else:
        # A model without a tokenizer (the tiny reference): DeepSeek's three control tokens, then bytes.
        for i, s in enumerate(["<｜begin▁of▁sentence｜>", "<｜end▁of▁sentence｜>", "<｜▁pad▁｜>"]):
            tokens[i] = s
            types[i] = 3
        for i, ch in enumerate(byte_level_alphabet()):
            tokens[3 + i] = ch
        merges = []
    for i in range(vocab_size):
        if tokens[i] is None:
            tokens[i] = f"[PAD{i}]"
            types[i] = 5
    writer.add_string("tokenizer.ggml.model", "gpt2")
    writer.add_string("tokenizer.ggml.pre", "joyai-llm")
    writer.add_array("tokenizer.ggml.tokens", gguf_io.GGUF_STRING, tokens)
    writer.add_array("tokenizer.ggml.token_type", gguf_io.GGUF_INT32, types)
    writer.add_array("tokenizer.ggml.merges", gguf_io.GGUF_STRING, merges)
    writer.add_uint32("tokenizer.ggml.bos_token_id", cfg.get("bos_token_id", 0))
    writer.add_uint32("tokenizer.ggml.eos_token_id", cfg.get("eos_token_id", 1))
    writer.add_uint32("tokenizer.ggml.padding_token_id", 2)
    writer.add_bool("tokenizer.ggml.add_bos_token", False)
    writer.add_bool("tokenizer.ggml.add_eos_token", False)


def byte_level_alphabet():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    table = dict(zip(bs, cs))
    return [chr(table[b]) for b in range(256)]


def parse_layers(spec, n_layers):
    if spec is None:
        return list(range(n_layers))
    a, _, b = spec.partition("-")
    first, last = int(a), int(b or a)
    if not 0 <= first <= last < n_layers:
        raise SystemExit(f"layers out of range: {spec}")
    return list(range(first, last + 1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--model", help="local checkpoint directory")
    src.add_argument("--hf-repo", help="Hugging Face repository to download the needed shards from")
    ap.add_argument("--download-dir", help="where --hf-repo files go")
    ap.add_argument("--delete-shards", action="store_true", help="remove each layer shard once converted")
    ap.add_argument("--out", required=True)
    ap.add_argument("--type", default="Q8_0", choices=["Q8_0", "Q4_0", "F16", "F32"],
                    help="type of the FP8 matrices (default Q8_0)")
    ap.add_argument("--expert-type", choices=["Q8_0", "Q4_0", "MXFP4", "F16", "F32"],
                    help="type of the routed experts (default: MXFP4 for FP4 experts, else --type)")
    ap.add_argument("--small-type", default="BF16", choices=["BF16", "F32"],
                    help="type of the BF16 matrices: compressors, indexer weights, router")
    ap.add_argument("--head-type", choices=["Q8_0", "F16", "BF16", "F32"],
                    help="type of the embedding and the head (default: Q8_0 for a quantized --type,"
                         " else --small-type)")
    ap.add_argument("--layers", help="contiguous decoder layers to include, e.g. 0-3 (default all)")
    ap.add_argument("--mtp", action="store_true", help="include the MTP layer")
    args = ap.parse_args()

    model_dir = args.model
    if args.hf_repo:
        if not args.download_dir:
            raise SystemExit("--hf-repo needs --download-dir")
        model_dir = args.download_dir
        download(args.hf_repo, model_dir, ["config.json", "model.safetensors.index.json"])
    with open(os.path.join(model_dir, "config.json")) as f:
        cfg = json.load(f)
    n_layers = cfg["num_hidden_layers"]
    layers = parse_layers(args.layers, n_layers)
    n_mtp = cfg.get("num_nextn_predict_layers", 0) if args.mtp else 0

    with open(os.path.join(model_dir, "model.safetensors.index.json")) as f:
        weight_map = json.load(f)["weight_map"]
    prefixes = [f"layers.{i}." for i in layers] + [f"mtp.{i}." for i in range(n_mtp)]
    shard_of_layer = {p: sorted({s for n, s in weight_map.items() if n.startswith(p)}) for p in prefixes}
    top_shards = sorted({weight_map[n] for n in TOP_NAMES})
    if args.hf_repo:
        download(args.hf_repo, model_dir, ["tokenizer.json", "tokenizer_config.json"] + top_shards)

    src = Source(model_dir, args.hf_repo)
    target = gguf_io.TYPE_BY_NAME[args.type]
    expert_type = gguf_io.TYPE_BY_NAME[args.expert_type] if args.expert_type else None
    plan = Plan(src, target, gguf_io.TYPE_BY_NAME[args.small_type])
    head_name = args.head_type or ("Q8_0" if args.type in ("Q8_0", "Q4_0") else args.small_type)
    head_type = gguf_io.TYPE_BY_NAME[head_name]
    plan.add("token_embd.weight", "embed.weight", head_type)
    plan.add("output_norm.weight", "norm.weight")
    plan.add("output.weight", "head.weight", head_type)
    for name in ("hc_head_fn", "hc_head_base", "hc_head_scale"):
        plan.add(TOP_NAMES[name], name)
    top_count = len(plan.items)

    ratios = cfg["compress_ratios"]
    n_hash = cfg["num_hash_layers"]
    out_ratios, layer_items = [], {}
    for out_index, prefix in enumerate(prefixes):
        is_mtp = prefix.startswith("mtp.")
        i = n_layers + int(prefix.split(".")[1]) if is_mtp else int(prefix.split(".")[1])
        g = f"blk.{out_index}."
        start = len(plan.items)
        for rel, gname in LAYER_NAMES.items():
            name = prefix + rel
            if rel == "ffn.gate.bias" and i < n_hash:
                continue  # hash-routed layers ignore the bias
            if src.has(name):
                plan.add(g + gname, name)
        if src.has(prefix + "ffn.gate.tid2eid"):
            plan.add_i32(g + "ffn_gate_tid2eid.weight", prefix + "ffn.gate.tid2eid")
        plan.add_experts(g, prefix + "ffn.experts.", cfg["n_routed_experts"], expert_type)
        if is_mtp:
            for rel, gname in MTP_NAMES.items():
                plan.add(g + gname, prefix + rel)
            plan.add_eh_proj(g + "nextn.eh_proj.weight", prefix + "e_proj.weight", prefix + "h_proj.weight")
        out_ratios.append(ratios[i] if i < len(ratios) else 0)
        layer_items[prefix] = (start, len(plan.items))

    blocks = len(prefixes)
    hash_count = max(0, min(n_hash, layers[-1] + 1) - layers[0])
    rs = cfg.get("rope_scaling") or {}
    w = gguf_io.GGUFWriter(args.out, ARCH)
    full = len(layers) == n_layers
    w.add_string("general.name", "DeepSeek-V4" if full else f"DeepSeek-V4 layers {layers[0]}-{layers[-1]}")
    w.add_uint32("general.file_type", {"F32": 0, "F16": 1, "Q4_0": 2, "Q8_0": 7}[args.type])
    k = ARCH + "."
    w.add_uint32(k + "block_count", blocks)
    if n_mtp:
        w.add_uint32(k + "nextn_predict_layers", n_mtp)
    w.add_uint32(k + "context_length", cfg["max_position_embeddings"])
    w.add_uint32(k + "embedding_length", cfg["hidden_size"])
    w.add_uint32(k + "embedding_length_out", cfg["hidden_size"] * cfg["hc_mult"])
    w.add_uint32(k + "vocab_size", cfg["vocab_size"])
    w.add_uint32(k + "attention.head_count", cfg["num_attention_heads"])
    w.add_uint32(k + "attention.head_count_kv", cfg.get("num_key_value_heads", 1))
    if rs:
        w.add_string(k + "rope.scaling.type", rs.get("rope_type", rs.get("type", "yarn")))
        w.add_float32(k + "rope.scaling.factor", rs["factor"])
        w.add_uint32(k + "rope.scaling.original_context_length", rs["original_max_position_embeddings"])
        w.add_float32(k + "rope.scaling.yarn_beta_fast", rs.get("beta_fast", 32))
        w.add_float32(k + "rope.scaling.yarn_beta_slow", rs.get("beta_slow", 1))
    w.add_float32(k + "rope.freq_base", cfg["rope_theta"])
    w.add_float32(k + "attention.layer_norm_rms_epsilon", cfg["rms_norm_eps"])
    w.add_uint32(k + "expert_count", cfg["n_routed_experts"])
    w.add_uint32(k + "expert_used_count", cfg["num_experts_per_tok"])
    w.add_uint32(k + "expert_gating_func", 4)  # sqrtsoftplus
    w.add_uint32(k + "attention.key_length", cfg["head_dim"])
    w.add_uint32(k + "attention.value_length", cfg["head_dim"])
    w.add_uint32(k + "rope.dimension_count", cfg["qk_rope_head_dim"])
    w.add_uint32(k + "attention.q_lora_rank", cfg["q_lora_rank"])
    w.add_uint32(k + "attention.sliding_window", cfg["sliding_window"])
    w.add_uint32(k + "expert_feed_forward_length", cfg["moe_intermediate_size"])
    w.add_uint32(k + "expert_shared_count", cfg["n_shared_experts"])
    w.add_float32(k + "expert_weights_scale", cfg["routed_scaling_factor"])
    w.add_bool(k + "expert_weights_norm", cfg["norm_topk_prob"])
    w.add_array(k + "swiglu_clamp_exp", gguf_io.GGUF_FLOAT32, [cfg["swiglu_limit"]] * blocks)
    w.add_array(k + "swiglu_clamp_shexp", gguf_io.GGUF_FLOAT32, [cfg["swiglu_limit"]] * blocks)
    w.add_uint32(k + "attention.indexer.head_count", cfg["index_n_heads"])
    w.add_uint32(k + "attention.indexer.key_length", cfg["index_head_dim"])
    w.add_uint32(k + "attention.indexer.top_k", cfg["index_topk"])
    w.add_uint32(k + "attention.output_group_count", cfg["o_groups"])
    w.add_uint32(k + "attention.output_lora_rank", cfg["o_lora_rank"])
    w.add_array(k + "attention.compress_ratios", gguf_io.GGUF_UINT32, out_ratios)
    w.add_float32(k + "attention.compress_rope_freq_base", cfg["compress_rope_theta"])
    w.add_uint32(k + "hyper_connection.count", cfg["hc_mult"])
    w.add_uint32(k + "hyper_connection.sinkhorn_iterations", cfg["hc_sinkhorn_iters"])
    w.add_float32(k + "hyper_connection.epsilon", cfg["hc_eps"])
    w.add_uint32(k + "hash_layer_count", hash_count)
    tokenizer_metadata(w, model_dir, cfg)

    for name, ne, t, _ in plan.items:
        w.add_tensor_info(name, ne, t)
    w.write_header()
    total = len(plan.items)

    def write(n0, n1):
        for n in range(n0, n1):
            name, ne, t, produce = plan.items[n]
            w.write_tensor_parts(name, produce())
            print(f"[{n + 1}/{total}] {name} {ne} {gguf_io.TYPE_NAMES[t]}", flush=True)

    write(0, top_count)
    for prefix in prefixes:
        if args.hf_repo:
            download(args.hf_repo, model_dir, shard_of_layer[prefix])
        write(*layer_items[prefix])
        if args.delete_shards and args.hf_repo:
            src.ckpt.close_unused()
            for shard in shard_of_layer[prefix]:
                if shard not in top_shards:
                    os.remove(os.path.join(model_dir, shard))
    w.close()
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
