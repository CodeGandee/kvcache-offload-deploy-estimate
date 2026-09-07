# KV-cache offload deployment estimates

This repository develops reproducible planning estimates for deploying KV-cache
offloading and sparse-attention techniques on different server topologies. It is
intended to accumulate additional hardware, topology, model, and scheduler cases.

The first case studies ShadowKV-style 1.56% cache selection at 128K and 256K context
on one or two 8×NVIDIA A800 80 GB servers. Kimi Code 2.7 and GLM-5.3 use PP8×TP2
across two 400 Gb/s InfiniBand-connected servers; DeepSeek V4 Flash uses two
four-GPU replicas on one server.

> These are analytical planning centers, not measured A800 benchmarks. Read the
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
```

If the repository was cloned without dependencies, initialize them with:

```bash
git submodule update --init --recursive
```

## Repository layout

- `docs/`: methodology, cases, sources, and the interactive report.
- `src/kvcache_offload_deploy_estimate/`: small auditable estimation formulas.
- `tests/`: formula and publication-integrity checks.
- `extern/tracked/`: pinned upstream research and model-code submodules.
- `extern/orphan/`: ignored local-only experiments.
- `context/`: durable development context, separate from public documentation.

The official model-weight formats are discussed in the estimate, but weights are
not stored in this repository. Hugging Face submodules are checked out with Git LFS
smudging disabled in the project checkout; their gitlinks pin metadata and source
revisions only.
