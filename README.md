# KV-cache offload deployment estimates

This repository develops reproducible planning estimates for deploying KV-cache
offloading and sparse-attention techniques on different server topologies. It is
intended to accumulate additional hardware, topology, model, and scheduler cases.

The first case studies ShadowKV-style 1.56% cache selection at 72K, 128K, and 256K
context on one or two 8×NVIDIA A800 80 GB servers. Kimi Code 2.7 and GLM-5.3 use
PP8×TP2 across two 400 Gb/s InfiniBand-connected servers; GLM-5.3-Flash uses one TP8
replica; DeepSeek V4 Flash uses two four-GPU replicas on one server.

The same report also contains a no-ShadowKV control. It keeps each model's complete
native growing attention state in HBM, retains native DSA/KDA/compressed-attention
behavior, and rejects the first request beyond a memory-only whole-request ceiling.

The current pipeline reads official model configs, derives roofline service times with
GenZ, serializes a 1–1,152-sequence sweep in LLMServingSim profile format, then adds the
ShadowKV selection, reconstruction, transfer, overlap, MTP, and residency events. The
extended sweep lets the full-resident control model high compressed-cache concurrency
without clamping. Its native-attention term is an explicit compute/HBM/dequant
roofline. The project contains no manually selected model-forward latency curve.

> These are simulation-assisted planning centers, not measured A800 benchmarks. Read the
> uncertainty section before using the numbers for capacity commitments.

## Current result

- [Live documentation](https://codegandee.github.io/kvcache-offload-deploy-estimate/)
- [Live interactive report](https://codegandee.github.io/kvcache-offload-deploy-estimate/cases/a800-pp8-tp2-shadowkv.html)
- [Case summary](docs/cases/a800-pp8-tp2.md)
- [Interactive standalone report](docs/cases/a800-pp8-tp2-shadowkv.html)
- [Estimation approach](docs/approach.md)
- [Reproduction guide](docs/reproducibility.md)

## Quick start

```bash
git clone --recurse-submodules https://github.com/CodeGandee/kvcache-offload-deploy-estimate.git
cd kvcache-offload-deploy-estimate
pixi install
pixi run check
pixi run docs
pixi run kv-estimate 1 8 16 32
pixi run kv-shadowkv-sim --samples 256
pixi run profiles
pixi run report-assets
```

If the repository was cloned without dependencies, initialize them with:

```bash
git submodule update --init --recursive
```

## Repository layout

- `docs/`: methodology, cases, sources, and the interactive report.
- `src/kvcache_offload_deploy_estimate/`: auditable formulas and the external
  LLMServingSim/ShadowKV trace model.
- `tests/`: formula and publication-integrity checks.
- `extern/tracked/`: pinned upstream research and model-code submodules.
- `extern/orphan/`: ignored local-only experiments.
- `context/`: durable development context, separate from public documentation.

The official model-weight formats are discussed in the estimate, but weights are
not stored in this repository. Hugging Face submodules are checked out with Git LFS
smudging disabled in the project checkout; their gitlinks pin metadata and source
revisions only.
