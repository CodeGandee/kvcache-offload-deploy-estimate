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
- An optimistic 80%-hit token-ahead case and a fetch-at-decode case with 60% reuse.

[Read the case summary](cases/a800-pp8-tp2.md) or open the
[interactive report](cases/a800-pp8-tp2-shadowkv.html).

## Interpretation

The estimates answer “what happens if the algorithm and custom runtime work as
specified?” They are not vendor benchmarks. Aggregate throughput and TPOT carry
roughly ±30–50% uncertainty in the token-ahead case and ±40–60% in the
fetch-at-decode case. Cold TTFT can vary by at least ±50%.

Future cases should reuse the documented method while replacing hardware bandwidth,
checkpoint placement, parallel topology, selector behavior, and calibration data.
