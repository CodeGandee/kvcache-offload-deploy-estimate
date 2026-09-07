# Data artifacts

The first case keeps its central curves and p10–p90 sensitivity ranges embedded in
the standalone HTML report so it remains portable. Regenerate the machine-readable
form on demand with `pixi run kv-shadowkv-sim --samples 256 --json`.

The A100 hardware-proxy calibration is stored as public-safe raw JSON at
`benchmarks/a100-sxm4/results/a100-sxm4-80gb-gpu2-3.json`. It includes repeated
sample distributions for HBM, H2D, P2P, FP8 conversion, and representative GEMMs;
the exact generator, CUDA kernel, Pixi manifest, and lock file sit beside it.
