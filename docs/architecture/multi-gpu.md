# Running a model across several GPUs

A model too large for one GPU can be split by layers across several, one pipeline stage per
device, as llama.cpp's `--split-mode layer` does it. Each GPU holds a contiguous range of layers
with their weights and key/value cache; only the hidden state moves between GPUs, once per stage
boundary per token. For one request at a time a split adds memory, not speed: the stages of a
token run one after the other.

## Quick start

```bash
./jitllm run --gpu --cuda --cuda-graphs --devices 0:0,0:1 --gpu-memory 21GB \
  -m Llama-3.3-70B-Instruct-Q4_0.gguf -c 2048 --prompt "Why is the sky blue?"
```

Llama-3.3-70B Q4_0 (40 GB) runs this way on two 24 GB NVIDIA A10s at about 10 tokens/s.

| Flag | Meaning |
| --- | --- |
| `--devices 0:0,0:1` | TornadoVM `backend:device` indices, one pipeline stage per device, in order. `tornado --devices` lists them. |
| `--tensor-split 41,39` | Each device's share of the layers, in proportion (default: equal). Give the last device a little less: it also holds the output projection. |
| `--split-transport nccl` | Send the hidden state GPU to GPU with NCCL (the default). |
| `--split-transport host` | Copy the hidden state through host memory instead; needs no NCCL. |
| `--gpu-memory 21GB` | Applies to **each** device. |

`--devices`, `--tensor-split` and `--split-transport` set the system properties
`jitllm.pipeline.devices`, `jitllm.pipeline.split` and `jitllm.pipeline.transport`, which a
direct Java launch can set itself.

## Requirements

- **NCCL transport** (the default):
  1. A TornadoVM SDK that ships `tornado-nccl` (`share/java/tornado/tornado-nccl-*.jar`).
  2. jitllm built with the `nccl` Maven profile: `./mvnw -Pnccl package -DskipTests`. A build
     without it refuses the split with "the NCCL pipeline transport is not in this build".
  3. `libnccl.so.2` on `LD_LIBRARY_PATH`, for example the `lib` directory of the `nvidia-nccl`
     wheel. Without it the run fails with "Unable to load NCCL".
- **Host transport**: nothing beyond a normal build; add `--split-transport host`.
- `--with-native-libraries` on a split also needs cuBLAS 12 on `LD_LIBRARY_PATH` (see the
  README's "Fast prompt processing on CUDA").

## Which models split

A split keeps single-token prefill by default, because not every family splits its batched
prefill; ask for batches with `--batch-prefill-size N`. Native libraries then follow the
single-device default (used where implemented and installed, `--no-native-libraries` to
decline).

| Family | Single-token decode | Batched prefill across devices |
| --- | --- | --- |
| Llama Q4_0 (e.g. Llama-3.3-70B) | yes | yes, with `--batch-prefill-size 256 --with-native-libraries` (int8 / cuBLAS layers) |
| Llama Q8_0 | yes | no: drop `--batch-prefill-size` to split the single-token plan |
| Qwen3.5 / Qwen3.8 (`qwen35`) | yes | yes, `--batch-prefill-size 256` (int8 tensor-core GEMMs on the quantized weights) |
| Qwen3.6 MoE (`qwen35moe`) | yes | yes, `--batch-prefill-size 256`; tokens are grouped by expert so each expert's rows run as one GEMM |
| Gemma 4 Q8_0 (e.g. 31B) | yes | yes, `--batch-prefill-size 256` |
| GLM-4.7-Flash (`deepseek2`) | yes | yes, `--batch-prefill-size 256` |

Any other family is refused when a split is asked for, naming the family. A family joins by
implementing `StagedForwardPlanComponents` (its layers for a range `[firstLayer, endLayer)`) and,
for batched prefill, the ranged `batchPrefillTransformerLayers` / `batchDecodeTransformerLayers`.

## How it works

```
stage 0 (device 0): embedding, layers [0, k1), send x ──┐
stage 1 (device 1): receive x, layers [k1, k2), ...  ◀──┘ ... final norm, logits
```

- **One plan per stage** (`TornadoVMMasterPlanPipeline`). Each stage builds its own
  TornadoVM execution plan, on its own device, from the family's plan components and a state of
  its own (`TornadoPlanProvider.stageState`). The first stage also runs the embedding, the last
  also the final norm and vocabulary projection. The class names no family.
- **Layer bounds** come from `DeviceSplit.layerBounds`: the shares in proportion, rounded to the
  family's `stageLayerAlignment()`.
- **Key/value cache per stage.** A family whose provider says `stageCacheHoldsOnlyItsLayers()`
  (Gemma 4, deepseek2) gives each stage a cache for its own layers only; otherwise each stage's
  state is built for the whole model and only its layers are written.
- **Kernels per device.** Each stage chooses its kernels for the device it runs on, not the
  first device, so mixed devices get the right variants. Packed weights are repacked on every
  stage.
- **Hand-off** (`PipelineTransport`). A stage's last graph sends the hidden state; the next
  stage's first graph receives it under the name of the activation graph it replaces, which is
  what the family's first layer consumes. `HostTransport` copies through host memory and runs
  the stages one after the other on the calling thread. `NcclTransport` (built only with
  `-Pnccl`, loaded by name) sends device to device and runs every stage on its own thread.
- **Overlapped prefill chunks.** In a batched prefill each stage runs its layers on the whole
  chunk and then hands the chunk's hidden states on. With NCCL the stages run concurrently, so
  the first device starts the next chunk while the next device still works on the previous one.
- **Sampling** happens on the last stage: the greedy token is read from the last stage's logits
  on the device.

## Troubleshooting

| Message | Cause | Fix |
| --- | --- | --- |
| `the NCCL pipeline transport is not in this build` | jitllm built without `-Pnccl` | Rebuild with `-Pnccl`, or use `--split-transport host` |
| `Unable to load NCCL ... libnccl.so.2` | NCCL library not found | Put the directory holding `libnccl.so.2` on `LD_LIBRARY_PATH` |
| `batched prefill is not split across devices` | The family splits only its single-token plan (e.g. Llama Q8_0) | Drop `--batch-prefill-size` |
| `batched prefill of a pipeline stage runs its projections through cuBLAS` | Llama Q4_0 stages that hold part of the model batch their projections in cuBLAS only | Add `--with-native-libraries` |
| `plan cannot be split across devices` | The family, or its weights' representation, has no stage layers | Drop `--devices` |
| `--devices entries are backend:device indices` | A device given as a bare number | Use `0:0,0:1`, as `tornado --devices` prints them |
| `--tensor-split has N entries for M devices` | Shares and devices differ in count | Give one share per device, or leave `--tensor-split` out |
| `GPUL-MEM-001` on one stage | That device's layers, cache or the output projection do not fit | Raise `--gpu-memory`, shift layers with `--tensor-split`, or shorten the context |
