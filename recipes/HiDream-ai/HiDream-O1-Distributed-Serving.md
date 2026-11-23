# HiDream-O1 Distributed Serving: Design and Progress

This document records the HiDream-O1-Image serving discussion, the changes
currently implemented on `dist_hidream`, their validation status, and a
measured-first roadmap for distributed inference work.

## Goals

- Keep expensive UiT computation on the high-performance GPU pool.
- Move suitable preprocessing and postprocessing to lower-cost CPU/GPU workers.
- Use asynchronous scheduling and compatible batching to reduce accelerator
  idle time and improve throughput.
- Measure latency, quality, memory, and communication costs before claiming
  performance improvements.

## Model and serving path

HiDream-O1 is a pixel-level unified transformer (UiT). The current vLLM-Omni
implementation has a shared Qwen3-VL-derived text Transformer, timestep
conditioning, and raw RGB patch embeddings.

For the text-to-image path:

1. The processor and tokenizer construct text tokens and position metadata.
2. The initial RGB noise is split into 32x32 patches.
3. Each RGB patch is projected from 3072 values through a 1024-wide bottleneck
   to the 4096-wide Transformer hidden size.
4. Text, timestep, and image patch representations are processed by the shared
   Transformer.
5. A final projection predicts RGB patch values. The scheduler iterates over
   the denoising steps; the patches are restored to an image at the end.

This is not a VAE pipeline: the current path does not use a VAE encoder or
decoder. The image postprocessing is patch reassembly and RGB conversion.

Implementation references:

- [HiDream-O1 pipeline](../../vllm_omni/diffusion/models/hidream_o1_image/pipeline_hidream_o1_image.py)
- [HiDream-O1 Transformer](../../vllm_omni/diffusion/models/hidream_o1_image/hidream_o1_image_transformer.py)
- [HiDream-O1 recipe](./HiDream-O1-Image.md)

## Implemented changes

### CFG conditional/unconditional batch

When classifier-free guidance (CFG) is enabled, the pipeline now tries to run
conditional and unconditional samples as batch size two in one Transformer
call:

- Concatenates token IDs, position IDs, image patch inputs, masks, and attention
  spans along their batch dimensions.
- Splits the predicted patches back into conditional and unconditional outputs.
- Applies the existing CFG velocity formula after the split.
- Falls back to two separate forwards when sample sequence shapes are
  incompatible.

The CFG-off path remains a single forward. This change reduces the number of
Python-level model calls per denoising step when batching is possible; actual
latency and throughput gains have not yet been measured on CUDA hardware.

### Static text-sample LRU cache

The pipeline caches the prepared static text sample used by a prompt and
resolution. Its key currently includes model path, model revision, prompt,
height, width, expected image-token count, dtype, and piecewise-attention
support. Cache entries are promoted on a hit and the least-recently-used entry
is removed when the entry-count limit (128) is exceeded.

The cache exposes hit, miss, entry, and capacity counters through
`get_text_sample_cache_stats()`. It caches prepared token/position/attention
metadata, not denoising outputs or per-step image predictions.

## Validation status

Validation completed locally without CUDA:

- Python syntax compilation for the changed pipeline and test module.
- CPU assertions for CFG batch tensor assembly, one-call output splitting, and
  matching-shape behavior.
- CPU assertions for static sample cache hits and misses.
- `git diff --check`.

The CFG model call in the focused assertion uses a mock model. This does not
validate a real Transformer forward, image quality, BF16/CUDA behavior, or
performance. Full pytest collection has also been blocked by the local
Windows/WSL dependency and vLLM build mismatch. No GPU is available locally.

Therefore:

- No CUDA/H100 accuracy result is available yet.
- No speedup, throughput gain, memory saving, or latency reduction is claimed.
- The previous performance percentages discussed informally are hypotheses,
  not benchmark results.

## Important cache limitation

The text-sample cache is bounded by entry count, not by bytes. Some cached
samples can contain device tensors, including a dense attention mask when the
attention backend does not support piecewise spans. A dense sequence mask grows
quadratically with sequence length, so 128 entries can consume substantial GPU
memory for high-resolution requests.

Before enabling this cache broadly in production, measure its memory footprint
and consider:

- A byte-budgeted cache rather than only an entry-count limit.
- Avoiding storage of reconstructible dense masks.
- Keeping appropriate static metadata on CPU and transferring it when needed.
- Cache eviction and synchronization behavior under concurrent requests.

## Distributed serving design discussed

The proposed serving split is three logical planes:

```text
Preprocess plane  ->  UiT compute plane  ->  Postprocess plane
```

### Preprocess plane

Suitable initial candidates include prompt templating/tokenization, static text
sample construction, position/attention metadata, and noise initialization.
These tasks can be batched independently, subject to prompt length and
resolution compatibility.

### UiT compute plane

Keep the repeated Transformer denoising forwards and scheduler-dependent image
state close to the high-cost GPUs. Timestep embedding and dynamic image patch
embedding should remain local initially unless measurements show their cost
justifies transfer to another worker.

### Postprocess plane

Patch reassembly, RGB conversion, and image encoding can be moved to CPU or
lower-cost GPU workers. This implementation has no VAE decode stage to offload.

Use explicit request and step identities, backpressure, bounded queues,
cancellation, timeout/error propagation, and dtype/shape validation at service
boundaries. Avoid transferring every Transformer-layer hidden state across
nodes.

## CFG batching and LLM-style prefill/decode separation

CFG conditional/unconditional samples are independent until their predictions
are combined, making batch size two a practical first optimization. Its
trade-off is higher activation and attention memory; the latency benefit is
hardware- and backend-dependent and needs CUDA profiling.

HiDream can borrow asynchronous scheduling ideas from LLM serving, such as
request queues, batching, backpressure, and overlap. A direct prefill/decode
split is not a drop-in design: each diffusion step processes the image-token
sequence again, rather than appending one token while reusing a stable
autoregressive KV cache. Static text KV reuse may be explored only after the
mixed causal/full attention behavior and cache validity are proven.

## Candidate optimization roadmap

### Locally verifiable

1. Continue CPU unit coverage for CFG batch assembly and output separation.
2. Add prompt-cache key isolation, eviction, and memory-budget tests.
3. Add compatibility checks for batching across prompt lengths, resolutions,
   dtypes, and attention backends.
4. Add request/step-state tests before attempting continuous batching.
5. Keep performance statements explicitly unverified until hardware runs exist.

### Requires a compatible Linux/CUDA environment

1. Run the actual HiDream model with CFG enabled and compare against the
   separate-forward baseline.
2. Measure per-step latency, end-to-end latency, throughput, peak/reserved
   memory, and GPU utilization.
3. Verify output quality using the repository's SSIM/PSNR baseline.
4. Profile attention, MLP, TP communication, and cache memory before choosing
   tensor/sequence/CFG parallelism.
5. Prototype external preprocess/postprocess workers only after measuring
   serialization and network costs.

Potential distributed techniques include tensor parallelism, sequence/ring
attention, CFG replica parallelism, data-parallel replicas, and compatible
request batching. Their usefulness depends on sequence length, batch size,
hardware topology, and communication overhead. Attention/MLP layer-by-layer
splitting across ordinary network links is not an initial recommendation
because hidden-state transfer may dominate.

## Current branch and commits

The changes are on `dist_hidream` in
[hii-gyh/vllm-omni](https://github.com/hii-gyh/vllm-omni/tree/dist_hidream):

- `b41e8d1c2` — `feat(hidream): batch CFG forward passes`
- `f50e4a133` — `perf(hidream): cache static text samples`

This document summarizes the current implementation and discussion; it does not
represent a CUDA performance or image-quality certification.
