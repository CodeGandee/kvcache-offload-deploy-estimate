# KV-cache offload deployment estimates

This project estimates raw TTFT, TPOT, aggregate output throughput, per-user output
throughput, and simultaneous-user capacity for KV-cache offloading deployments.
Each case records its hardware, topology, checkpoint storage, runtime precision,
cache policy, scheduler assumptions, equations, calibration evidence, and uncertainty.

## Current case

The first published case covers:

- 72K, 128K, and 256K input contexts.
- One or two servers, each with 8×NVIDIA A800 80 GB.
- Kimi Code 2.7 and GLM-5.3 on two servers using PP8×TP2.
- GLM-5.3-Flash on one server using TP8; only its 11 sparse-attention layers use
  the hypothetical ShadowKV path.
- DeepSeek V4 Flash as two four-GPU replicas on one server.
- Official compressed model weights, fused conversion, and BF16 compute.
- BF16 or FP8 exact-KV storage.
- ShadowKV-style 1.56% cache selection.
- A common HBM-only OOM admission boundary for every case, with system-RAM capacity
  unbounded and finite host-transfer bandwidth retained in latency.
- Pure TP with DCP1: MLA/latent history, native sparse-index rows, ShadowKV state, and
  direct-copy H2D payload are replicated per TP rank; PP alone partitions layers.
- An advisory token-ahead oracle with 80% recall/precision and a fetch-at-decode case,
  both with 60% temporal reuse.
- A no-ShadowKV control that preserves native attention/cache compression, keeps all
  growing state in HBM, and rejects new requests at the memory-only admission limit.
- Separate native-MTP plots for one and two mean consecutively accepted draft tokens.
- Whole-layer exact-KV HBM residency from 0% through 100%, with a one-user/10%-step
  load slider and explicit HBM-feasibility markers.
- Fixed-format GLM/V4 index caches and GLM-Flash's TP-sharded mixed-precision KDA
  state, accounted separately from the main BF16/FP8 cache toggle.
- GenZ model-core rooflines generated from official model configs, stored precision,
  and an A100-SXM4 hardware envelope, then consumed as LLMServingSim profiles.
- A100-SXM4 measurements of HBM, PCIe sharing, NVLink/P2P, BF16 GEMM MBU, and fused
  FP8 conversion, used as an explicitly labeled A800 Ampere proxy.

[Read the case summary](cases/a800-pp8-tp2.md) or open the
[interactive report](cases/a800-pp8-tp2-shadowkv.html).

## Interpretation

The estimates answer “what happens if the algorithm and custom runtime work as
specified?” They are not vendor benchmarks. The generated p10–p90 ranges vary
explicit event timings and bandwidths; they are sensitivity intervals, not confidence
intervals. The model-core profiles are not full-model measurements. A100→A800
architecture/topology mismatch and cold TTFT can still vary by at least ±50%.

Future cases should reuse the documented method while replacing hardware bandwidth,
checkpoint placement, parallel topology, selector behavior, and calibration data.
