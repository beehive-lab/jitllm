"""Convert a GLM-5.3-Flash safetensors checkpoint to a llama.cpp glm5-next GGUF file.

The converter reads tensors through model.safetensors.index.json and opens only
the shards that hold the requested layers, so a slice of the model can be
converted without the whole checkpoint on disk. With --hf-repo it downloads just
those shards (plus config and tokenizer files) first. The output has llama.cpp's
glm5-next metadata, tensor names and layouts, so llama.cpp and jitllm load the
same file; a slice is written as a model of fewer blocks.

FP8 (E4M3, 128x128 block scales), BF16 and F32 inputs are supported. Large
matrices are written as --type (Q8_0, Q4_0, F16 or F32); norms, biases, gates and
the mHC parameters stay F32. Each layer's routed experts are one tensor per
projection, as llama.cpp stores them.

Examples:
    # first four layers (three KDA and one sparse-attention layer) as Q8_0
    python -I convert.py --hf-repo zai-org/GLM-5.3-Flash --download-dir ~/glm53 \\
        --layers 0-3 --type Q8_0 --out ~/models/glm53-l0-3-Q8_0.gguf
    # whole text model with the MTP layer
    python -I convert.py --model ~/glm53 --type Q4_0 --mtp --out ~/models/glm53-Q4_0.gguf
"""

import argparse
import json
import os
import re
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gguf_io  # noqa: E402

ARCH = "glm5-next"
PREFIX = "model.language_model."

# Tensors kept in F32 whatever the target type (small, or precision-sensitive), as llama.cpp's
# converter keeps them.
F32_PATTERNS = [
    r"\.hc_(attn|ffn)_(fn|base|scale)\.weight$",
    r"\.ffn_gate_inp\.weight$",
    r"\.exp_probs_b\.bias$",
    r"\.ssm_conv1d_[qkv]\.weight$",
    r"\.ssm_dt\.bias$",
    r"\.ssm_a$",
    r"\.indexer_compressor_(ape|gate)\.weight$",
    r"\.indexer\.proj\.weight$",
    r"norm\.(weight|bias)$",
]

KDA_NAMES = {
    "q_proj.weight": "attn_q.weight",
    "k_proj.weight": "attn_k.weight",
    "v_proj.weight": "attn_v.weight",
    "o_proj.weight": "attn_output.weight",
    "q_conv1d.weight": "ssm_conv1d_q.weight",
    "k_conv1d.weight": "ssm_conv1d_k.weight",
    "v_conv1d.weight": "ssm_conv1d_v.weight",
    "f_a_proj.weight": "ssm_f_a.weight",
    "f_b_proj.weight": "ssm_f_b.weight",
    "dt_bias": "ssm_dt.bias",
    "A_log": "ssm_a",
    "b_proj.weight": "ssm_beta.weight",
    "g_a_proj.weight": "ssm_g_a.weight",
    "g_b_proj.weight": "ssm_g_b.weight",
    "o_norm.weight": "ssm_norm.weight",
}

DSA_NAMES = {
    "q_a_proj.weight": "attn_q_a.weight",
    "q_a_layernorm.weight": "attn_q_a_norm.weight",
    "q_b_proj.weight": "attn_q_b.weight",
    "kv_a_proj_with_mqa.weight": "attn_kv_a_mqa.weight",
    "kv_a_layernorm.weight": "attn_kv_a_norm.weight",
    "o_proj.weight": "attn_output.weight",
    "indexer.wq_b.weight": "indexer.attn_q_b.weight",
    "indexer.wk.weight": "indexer.attn_k.weight",
    "indexer.k_norm.weight": "indexer.k_norm.weight",
    "indexer.k_norm.bias": "indexer.k_norm.bias",
    "indexer.weights_proj.weight": "indexer.proj.weight",
    "indexer.index_kpool_compress_ape": "indexer_compressor_ape.weight",
    "indexer.index_kpool_compress_gate": "indexer_compressor_gate.weight",
}

LAYER_NAMES = {
    "input_layernorm.weight": "attn_norm.weight",
    "post_attention_layernorm.weight": "ffn_norm.weight",
    "hc_attn_fn": "hc_attn_fn.weight",
    "hc_attn_base": "hc_attn_base.weight",
    "hc_attn_scale": "hc_attn_scale.weight",
    "hc_ffn_fn": "hc_ffn_fn.weight",
    "hc_ffn_base": "hc_ffn_base.weight",
    "hc_ffn_scale": "hc_ffn_scale.weight",
    "mlp.gate_proj.weight": "ffn_gate.weight",
    "mlp.up_proj.weight": "ffn_up.weight",
    "mlp.down_proj.weight": "ffn_down.weight",
    "mlp.gate.weight": "ffn_gate_inp.weight",
    "mlp.gate.e_score_correction_bias": "exp_probs_b.bias",
    "mlp.shared_experts.gate_proj.weight": "ffn_gate_shexp.weight",
    "mlp.shared_experts.up_proj.weight": "ffn_up_shexp.weight",
    "mlp.shared_experts.down_proj.weight": "ffn_down_shexp.weight",
    "eh_proj.weight": "nextn.eh_proj.weight",
    "enorm.weight": "nextn.enorm.weight",
    "hnorm.weight": "nextn.hnorm.weight",
    "shared_head.norm.weight": "nextn.shared_head_norm.weight",
}

EXPERT_PROJ = (("gate_proj", "ffn_gate_exps"), ("up_proj", "ffn_up_exps"), ("down_proj", "ffn_down_exps"))


def parse_layers(spec, n_layers):
    if spec is None:
        return list(range(n_layers))
    out = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    bad = [i for i in out if not 0 <= i < n_layers]
    if bad:
        raise SystemExit(f"layers out of range: {bad}")
    return sorted(set(out))


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


class Plan:
    """The tensors to write: (gguf name, ne, type, producer of byte pieces)."""

    def __init__(self, ckpt, target):
        self.ckpt = ckpt
        self.target = target
        self.items = []

    def type_for(self, name, ne):
        if len(ne) == 1 or any(re.search(p, name) for p in F32_PATTERNS):
            return gguf_io.F32
        if ne[0] % 32:
            return gguf_io.F32
        if self.target == gguf_io.Q4_0 and name in ("token_embd.weight", "output.weight"):
            return gguf_io.Q8_0
        return self.target

    def add(self, gguf_name, hf_name, reshape=None, transform=None):
        shape = self.ckpt.shape(hf_name)
        if reshape is not None:
            shape = reshape(shape)
        ne = list(reversed(shape))
        t = self.type_for(gguf_name, ne)

        def produce():
            x = self.ckpt.tensor_f32(hf_name)
            if transform is not None:
                x = transform(x)
            yield gguf_io.encode(x.reshape(shape), t)

        self.items.append((gguf_name, ne, t, produce))

    def add_kv_b(self, gguf_prefix, hf_name, heads, nope, value):
        """Splits kv_b_proj the way llama.cpp does: k_b [heads, rank, nope], v_b [heads, value, rank]."""
        rank = self.ckpt.shape(hf_name)[1]
        for gname, ne, part in ((gguf_prefix + "attn_k_b.weight", [nope, rank, heads], "k"),
                                (gguf_prefix + "attn_v_b.weight", [rank, value, heads], "v")):
            t = self.type_for(gname, ne)

            def produce(part=part, t=t):
                kv = self.ckpt.tensor_f32(hf_name).reshape(heads, nope + value, rank)
                x = kv[:, :nope, :].transpose(0, 2, 1) if part == "k" else kv[:, nope:, :]
                yield gguf_io.encode(np.ascontiguousarray(x), t)

            self.items.append((gname, ne, t, produce))

    def add_experts(self, gguf_prefix, hf_prefix, n_experts):
        """One tensor per projection over every expert, [experts, rows, cols], as llama.cpp stores them."""
        for proj, gname in EXPERT_PROJ:
            first = f"{hf_prefix}experts.0.{proj}.weight"
            rows, cols = self.ckpt.shape(first)
            name = f"{gguf_prefix}{gname}.weight"
            ne = [cols, rows, n_experts]
            t = self.type_for(name, ne)

            def produce(proj=proj):
                for e in range(n_experts):
                    yield gguf_io.encode(self.ckpt.tensor_f32(f"{hf_prefix}experts.{e}.{proj}.weight"), t)

            self.items.append((name, ne, t, produce))


def tokenizer_metadata(writer, model_dir, vocab_size):
    path = os.path.join(model_dir, "tokenizer.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            tok = json.load(f)
        vocab = tok["model"]["vocab"]
        merges = [m if isinstance(m, str) else " ".join(m) for m in tok["model"]["merges"]]
        tokens = [None] * vocab_size
        types = [1] * vocab_size
        for text, i in vocab.items():
            tokens[i] = text
        for added in tok["added_tokens"]:
            tokens[added["id"]] = added["content"]
            types[added["id"]] = 3 if added["special"] else 4
    else:
        # The tiny reference model has no tokenizer: bytes plus GLM's special tokens.
        tokens = [None] * vocab_size
        types = [1] * vocab_size
        for i, ch in enumerate(byte_level_alphabet()):
            tokens[i] = ch
        specials = ["<|endoftext|>", "[MASK]", "[gMASK]", "[sMASK]", "<sop>", "<eop>", "<|system|>",
                    "<|user|>", "<|assistant|>", "<|observation|>", "<think>", "</think>"]
        for j, s in enumerate(specials):
            tokens[256 + j] = s
            types[256 + j] = 3
        merges = []
    for i in range(vocab_size):
        if tokens[i] is None:
            tokens[i] = f"[PAD{i}]"
            types[i] = 5
    ids = {t: i for i, t in enumerate(tokens)}
    writer.add_string("tokenizer.ggml.model", "gpt2")
    writer.add_string("tokenizer.ggml.pre", "glm4")
    writer.add_array("tokenizer.ggml.tokens", gguf_io.GGUF_STRING, tokens)
    writer.add_array("tokenizer.ggml.token_type", gguf_io.GGUF_INT32, types)
    writer.add_array("tokenizer.ggml.merges", gguf_io.GGUF_STRING, merges)
    for key, text in (("eos_token_id", "<|endoftext|>"), ("padding_token_id", "<|endoftext|>"),
                      ("bos_token_id", "[gMASK]"), ("eot_token_id", "<|user|>"),
                      ("eom_token_id", "<|observation|>")):
        if text in ids:
            writer.add_uint32(f"tokenizer.ggml.{key}", ids[text])
    template = os.path.join(model_dir, "chat_template.jinja")
    if os.path.exists(template):
        with open(template, encoding="utf-8") as f:
            writer.add_string("tokenizer.chat_template", f.read())


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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--model", help="local checkpoint directory")
    src.add_argument("--hf-repo", help="Hugging Face repository to download the needed shards from")
    ap.add_argument("--download-dir", help="where --hf-repo files go")
    ap.add_argument("--out", required=True)
    ap.add_argument("--type", default="Q8_0", choices=["Q8_0", "Q4_0", "F16", "F32"])
    ap.add_argument("--layers", help="decoder layers to include, e.g. 0-3 or 0,3,7 (default all)")
    ap.add_argument("--mtp", action="store_true", help="include the MTP (next-token prediction) layer")
    ap.add_argument("--context-length", type=int, help="override the context length in the metadata")
    args = ap.parse_args()

    model_dir = args.model
    if args.hf_repo:
        if not args.download_dir:
            raise SystemExit("--hf-repo needs --download-dir")
        model_dir = args.download_dir
        download(args.hf_repo, model_dir, ["config.json", "model.safetensors.index.json"])
    with open(os.path.join(model_dir, "config.json")) as f:
        full = json.load(f)
    cfg = full.get("text_config", full)
    n_layers = cfg["num_hidden_layers"]
    layers = parse_layers(args.layers, n_layers)
    mtp_layers = list(range(n_layers, n_layers + cfg.get("num_nextn_predict_layers", 0))) if args.mtp else []

    if args.hf_repo:
        with open(os.path.join(model_dir, "model.safetensors.index.json")) as f:
            wm = json.load(f)["weight_map"]
        wanted = {f"{PREFIX}layers.{i}." for i in layers + mtp_layers}
        shards = {s for n, s in wm.items()
                  if not n.startswith("model.visual.")
                  and (not n.startswith(PREFIX + "layers.") or any(n.startswith(w) for w in wanted))}
        download(args.hf_repo, model_dir, ["tokenizer.json", "tokenizer_config.json", "chat_template.jinja"]
                 + sorted(shards))

    ckpt = gguf_io.Checkpoint(model_dir)
    target = gguf_io.TYPE_BY_NAME[args.type]
    plan = Plan(ckpt, target)

    plan.add("token_embd.weight", PREFIX + "embed_tokens.weight")
    plan.add("output_norm.weight", PREFIX + "norm.weight")
    plan.add("output.weight", "lm_head.weight")

    la_heads = cfg["linear_attn_config"]["num_heads"]
    layer_types = cfg["layer_types"]
    mlp_types = cfg.get("mlp_layer_types") or (["dense"] * cfg["first_k_dense_replace"]
                                                + ["sparse"] * (n_layers - cfg["first_k_dense_replace"]))
    out_layers = layers + mtp_layers
    for out_index, i in enumerate(out_layers):
        is_mtp = i >= n_layers
        hf = f"{PREFIX}layers.{i}."
        g = f"blk.{out_index}."
        for rel, gname in LAYER_NAMES.items():
            if ckpt.has(hf + rel):
                plan.add(g + gname, hf + rel)
        kda = not is_mtp and layer_types[i] == "linear_attention"
        for rel, gname in (KDA_NAMES if kda else DSA_NAMES).items():
            name = hf + "self_attn." + rel
            if not ckpt.has(name):
                raise SystemExit(f"missing tensor {name}")
            if rel == "kv_b_proj.weight":
                continue
            if rel.endswith("_conv1d.weight"):
                # ggml [d_conv, 1, d_inner, 1]: the same bytes as [d_inner, d_conv]
                plan.add(g + gname, name, reshape=lambda s: [1, s[0], 1, s[-1]])
            elif rel == "A_log":
                heads = la_heads
                plan.add(g + gname, name, reshape=lambda s: [heads],
                         transform=lambda x: -np.exp(x.reshape(-1)[:heads]))
            else:
                plan.add(g + gname, name)
        if not kda:
            plan.add_kv_b(g, hf + "self_attn.kv_b_proj.weight", cfg["num_attention_heads"],
                          cfg["qk_nope_head_dim"], cfg["v_head_dim"])
        if is_mtp or mlp_types[i] == "sparse":
            plan.add_experts(g, hf + "mlp.", cfg["n_routed_experts"])

    la = cfg["linear_attn_config"]
    dense = [mlp_types[i] == "dense" for i in layers]
    lead = dense.index(False) if False in dense else len(dense)
    if any(dense[lead:]):
        raise SystemExit("a glm5-next file has its dense layers first; this slice does not")
    blocks = len(out_layers)
    w = gguf_io.GGUFWriter(args.out, ARCH)
    w.add_string("general.name", "GLM-5.3-Flash" if len(layers) == n_layers else
                 f"GLM-5.3-Flash layers {args.layers}")
    w.add_uint32("general.file_type", {"F32": 0, "F16": 1, "Q4_0": 2, "Q8_0": 7}[args.type])
    k = ARCH + "."
    w.add_uint32(k + "block_count", blocks)
    if mtp_layers:
        w.add_uint32(k + "nextn_predict_layers", len(mtp_layers))
    w.add_uint32(k + "context_length", args.context_length or cfg["max_position_embeddings"])
    w.add_uint32(k + "embedding_length", cfg["hidden_size"])
    w.add_uint32(k + "feed_forward_length", cfg["intermediate_size"])
    w.add_uint32(k + "vocab_size", cfg["vocab_size"])
    w.add_uint32(k + "attention.head_count", cfg["num_attention_heads"])
    w.add_array(k + "attention.head_count_kv", gguf_io.GGUF_UINT32,
                [0 if layer_types[i] == "linear_attention" else 1 for i in layers] + [1] * len(mtp_layers))
    w.add_float32(k + "attention.layer_norm_rms_epsilon", cfg["rms_norm_eps"])
    w.add_float32(k + "attention.layer_norm_epsilon", 1e-6)
    w.add_uint32(k + "ssm.conv_kernel", la["short_conv_kernel_size"])
    w.add_uint32(k + "kda.head_dim", la["head_dim"])
    w.add_float32(k + "kda.gate_lower_bound", la.get("gate_lower_bound", -5.0))
    w.add_uint32(k + "attention.q_lora_rank", cfg["q_lora_rank"])
    w.add_uint32(k + "attention.kv_lora_rank", cfg["kv_lora_rank"])
    w.add_uint32(k + "rope.dimension_count", cfg["qk_rope_head_dim"])
    w.add_uint32(k + "attention.key_length", cfg["kv_lora_rank"] + cfg["qk_rope_head_dim"])
    w.add_uint32(k + "attention.value_length", cfg["kv_lora_rank"])
    w.add_uint32(k + "attention.key_length_mla", cfg["qk_nope_head_dim"] + cfg["qk_rope_head_dim"])
    w.add_uint32(k + "attention.value_length_mla", cfg["v_head_dim"])
    w.add_uint32(k + "attention.indexer.head_count", cfg["index_n_heads"])
    w.add_uint32(k + "attention.indexer.key_length", cfg["index_head_dim"])
    w.add_uint32(k + "attention.indexer.top_k", cfg["index_topk"])
    w.add_uint32(k + "attention.indexer.kpool", cfg["index_kpool"])
    w.add_bool(k + "attention.indexer.kpool_select_tail", cfg.get("index_kpool_always_select_tail", True))
    if cfg.get("indexer_types") is not None:
        w.add_array(k + "attention.indexer.types", gguf_io.GGUF_BOOL,
                    [cfg["indexer_types"][i] == "full" for i in layers])
    w.add_uint32(k + "hyper_connection.count", cfg["hc_mult"])
    w.add_uint32(k + "hyper_connection.sinkhorn_iterations", cfg["hc_sinkhorn_iters"])
    w.add_float32(k + "hyper_connection.epsilon", cfg["hc_eps"])
    w.add_uint32(k + "leading_dense_block_count", lead)
    w.add_uint32(k + "expert_feed_forward_length", cfg["moe_intermediate_size"])
    w.add_uint32(k + "expert_count", cfg["n_routed_experts"])
    w.add_uint32(k + "expert_used_count", cfg["num_experts_per_tok"])
    w.add_uint32(k + "expert_shared_count", cfg["n_shared_experts"])
    w.add_uint32(k + "expert_gating_func", 2)  # sigmoid
    w.add_uint32(k + "expert_group_count", cfg.get("n_group", 1))
    w.add_uint32(k + "expert_group_used_count", cfg.get("topk_group", 1))
    w.add_float32(k + "expert_weights_scale", cfg["routed_scaling_factor"])
    w.add_bool(k + "expert_weights_norm", cfg["norm_topk_prob"])
    if cfg.get("swiglu_limit") is not None:
        w.add_array(k + "swiglu_clamp_exp", gguf_io.GGUF_FLOAT32, [cfg["swiglu_limit"]] * blocks)
        w.add_array(k + "swiglu_clamp_shexp", gguf_io.GGUF_FLOAT32, [cfg["swiglu_limit"]] * blocks)
    tokenizer_metadata(w, model_dir, cfg["vocab_size"])

    for name, ne, t, _ in plan.items:
        w.add_tensor_info(name, ne, t)
    w.write_header()
    total = len(plan.items)
    for n, (name, ne, t, produce) in enumerate(plan.items, 1):
        w.write_tensor_parts(name, produce())
        print(f"[{n}/{total}] {name} {ne} {gguf_io.TYPE_NAMES[t]}", flush=True)
    w.close()
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
