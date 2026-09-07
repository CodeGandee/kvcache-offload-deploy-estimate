# KV-cache offload deployment estimates

This project estimates raw TTFT, TPOT, aggregate output throughput, per-user output
throughput, and simultaneous-user capacity for KV-cache offloading deployments.
Each case records its hardware, topology, checkpoint storage, runtime precision,
cache policy, scheduler assumptions, equations, calibration evidence, and uncertainty.

## Current case

The first published case covers:

- 128K and 256K input contexts.
- One or two servers, each with 8×NVIDIA A800 80 GB.
- Kimi Code 2.7 and GLM-5.3 on two servers using PP8×TP2.
- DeepSeek V4 Flash as two four-GPU replicas on one server.
- Official compressed model weights, fused conversion, and BF16 compute.
- BF16 or FP8 exact-KV storage.
- ShadowKV-style 1.56% cache selection.
- An advisory token-ahead oracle with 80% recall/precision and a fetch-at-decode case,
  both with 60% temporal reuse.

[Read the case summary](cases/a800-pp8-tp2.md) or open the
[interactive report](cases/a800-pp8-tp2-shadowkv.html).

## Interpretation

The estimates answer “what happens if the algorithm and custom runtime work as
specified?” They are not vendor benchmarks. The generated p10–p90 ranges vary
explicit event timings and bandwidths; they are sensitivity intervals, not confidence
intervals. Architecture mismatch and cold TTFT can still vary by at least ±50%.

Future cases should reuse the documented method while replacing hardware bandwidth,
checkpoint placement, parallel topology, selector behavior, and calibration data.
