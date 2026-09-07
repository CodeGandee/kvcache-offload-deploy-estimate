# A100-SXM4 calibration

This benchmark uses synthetic tensors, so it does not require model checkpoints. It
measures the hardware and kernel quantities used by the ShadowKV deployment model:

- copy-engine and SM-kernel HBM bandwidth;
- pinned host-to-device bandwidth for one and two GPUs;
- GPU-to-GPU peer-copy bandwidth and latency;
- FP8 E4M3 to BF16 conversion throughput on Ampere;
- representative BF16 GEMM shapes for PP8×TP2 decode stages.

Keep the environment and package cache on a local NVMe data disk and restrict
execution to physical GPUs 2 and 3. On the profiled host, both GPUs were attached to
NUMA node 0:

```bash
cd /path/on/local-data-disk/kvcache-offload-calibration-a100
PIXI_CACHE_DIR=/path/on/local-data-disk/.cache/pixi \
  CUDA_VISIBLE_DEVICES=2,3 \
  numactl --cpunodebind=0 --membind=0 \
  ~/.pixi/bin/pixi run calibrate
```

The JSON result is checked into `results/` together with the exact Pixi lock file.
It intentionally omits the private hostname and GPU UUIDs.

Print the central statistics from the repeated raw samples with:

```bash
pixi run summary
```

The run calibrates hardware primitives, not full models. In particular, its BF16
GEMMs do not include the official FP8/INT4/FP4 weight-unpack kernels, MoE routing,
attention, collectives, pipeline bubbles, or framework launch gaps. The report uses
the H2D, TP2-pair, P2P, and fused FP8-KV conversion measurements directly and treats
the GEMM MBU result as a plausibility check.
