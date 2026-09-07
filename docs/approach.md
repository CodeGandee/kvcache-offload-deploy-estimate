# Estimation approach

This repository uses a reproducible three-part pipeline:

1. official Hugging Face configs define each model's dimensions and attention type;
2. GenZ converts operator work and the measured A100 hardware envelope into
   decode/prefill roofline timings;
3. generated timings are written in LLMServingSim's profile-bundle format and read
   through LLMServingSim's production interpolation path before external ShadowKV
   events or the no-ShadowKV native-attention roofline are added.

It is still a simulation. No frontier-model weights were downloaded or executed.

## Models and placements

| Target | Official stored format | Active path | Placement | ShadowKV layers |
|---|---:|---:|---|---:|
| Kimi Code 2.7 | 595.2 GB, native INT4 | 32B | PP8×TP2, two nodes | 61 |
| GLM-5.3 | 755.6 GB, FP8 plus exclusions | 40B | PP8×TP2, two nodes | 78 |
| GLM-5.3-Flash | 328.3 GB, FP8 plus exclusions | 18B | TP8, one node | 11 of 45 |
| DeepSeek V4 Flash | 159.6 GB, FP4 experts + FP8 remainder | 13B | two TP4 replicas | 21 of 43 |

GLM-5.3-Flash is modeled separately. Its official config has 34 KDA
linear-attention layers and 11 DSA sparse-attention layers. Linear-attention recurrent
and convolution states remain in HBM; only the 11 DSA layers use the hypothetical
ShadowKV host-offload path.

The tracked Hugging Face repositories contain source/config metadata only. Git LFS
smudge is disabled, so model weight payloads are not checked out.

## Hardware inputs

The A100-SXM4 GPU 2+3 calibration contributes:

- 1,765 GB/s streaming HBM;
- 51.6% median BF16 GEMM memory-bandwidth utilization, with a measured
  30.7–61.8% range;
- 455 Gvalue/s fused E4M3 block-scale conversion to BF16;
- 22.0 GB/s long-transfer H2D to one GPU;
- 25.4 GB/s aggregate H2D for both GPUs in a TP2 pair;
- 0.031 ms small P2P latency and 274 GB/s large P2P bandwidth.

The deployment retains 180 GB/s usable host DRAM per node and 40 GB/s usable
one-way payload over 400 Gb/s InfiniBand as assumptions.

GenZ receives the measured HBM ceiling and a kernel-memory efficiency chosen so
effective weight bandwidth equals the measured 51.6% of the A100's 2,039 GB/s
nameplate:

\[
\eta_M=\frac{0.516\times2039}{1765}\approx0.596,
\qquad BW_{\mathrm{effective}}\approx1052\ \mathrm{GB/s}.
\]

BF16 peak is 312 TFLOP/s. Central compute efficiency is 40%, the GenZ paper's
profiling-derived A100 prior.

## GenZ model-forward profile

For aggregate operator \(j\), GenZ supplies its standard max-of-roofs execution
time. The adapter adds official stored precision and the measured conversion roof:

\[
t_j=\max\!\left(
  \frac{F_j}{\eta_C P_{\mathrm{BF16}}},
  \frac{B_{a,j}+B_{w,j}+B_{o,j}}{\eta_M BW_{\mathrm{HBM}}},
  \frac{W_j}{R_{\mathrm{dequant}}}
\right).
\]

Arithmetic is BF16. Weight bytes use official FP8/INT4/FP4 storage. The maximum
assumes successful fused conversion, HBM read, and matmul consumption. An unfused
implementation would add conversion time and be slower.

The active-parameter totals come from the model releases. Exact config dimensions
split each active path into shared/attention/LM-head and routed-expert work. For a
batch of \(C\) sequences with \(k\) choices among \(E\) experts, balanced routing
touches an expected

\[
E_{\mathrm{active}}(C)=E\left[1-\left(1-\frac1E\right)^{kC}\right].
\]

Shared weights are read once per batch. Routed-expert traffic grows with
\(E_{\mathrm{active}}\), while expert arithmetic grows with \(kC\). This replaces
the former hand-set load curves and explains why Kimi/GLM per-user performance
degrades at high concurrency.

Tensor-parallel collectives use a GenZ ring-style latency/bandwidth expression with
the measured P2P inputs. Two activation reductions per transformer block are
included. No cross-node expert parallelism is modeled.

## LLMServingSim bridge

The GenZ sweep covers 1–1,152 simultaneous sequences and is written to:

```text
data/profiles/llmservingsim/A100-SXM4-80GB/<model>/official-dequant-bf16/
├── meta.yaml
└── tp<N>/per_sequence.csv
```

The CSV uses LLMServingSim's documented schema:

```text
layer,sequences,time_us
model_core,1,...
model_core_stage_0,1,...
model_core,2,...
```

The estimator calls pinned upstream `_lookup_per_sequence`, which builds the table
and interpolates it exactly as LLMServingSim does. Total rows support inspection;
stage rows drive PP scheduling. The stage split follows the upstream vLLM-compatible
pipeline partitioner, with distributed block work proportional to the assigned block
count and the vocabulary projection on the last stage. ShadowKV events remain external
because tiered-KV block recall does not represent low-rank reconstruction or landmarks.

GenZ and LLMServingSim do different jobs: GenZ produces the missing
hardware/model service-time table; LLMServingSim consumes it using its serving
profile contract and supplies pipeline semantics.

## ShadowKV events

For context \(N\), selected fraction \(\alpha=1/64\), cache-bearing layers \(L\),
cached width \(d_c\), TP degree \(G\), and stored bytes/value \(b\):

\[
S(N)=\lceil\alpha N\rceil,
\qquad
B_{\mathrm{sel,GPU}}=\frac{S(N)Ld_cb}{G}.
\]

After 60% temporal reuse, the central 80%-recall/80%-precision token-ahead oracle
prefetches 0.40 of the selected set and leaves 0.08 just in time:

\[
f_{\mathrm{prefetch}}=(1-r)\frac{R_o}{P_o}=0.40,
\qquad
f_{\mathrm{JIT}}=(1-r)(1-R_o)=0.08.
\]

Prefetch overlaps the preceding token's model-forward window. The current token
still verifies landmarks. Fetch-at-decode has no background prefetch and exposes
the full 40% post-reuse miss set.

For a cache-bearing block:

\[
t_{\ell}=t_{\mathrm{select},\ell}
+\max(t_{K,\ell},t_{V,\ell}+t_{\mathrm{dequant},\ell})
+t_{\mathrm{attention},\ell}.
\]

ShadowKV Table 13 grounds selector/materialization centers. Reusing its Llama
per-block measurements on these architectures is a major extrapolation.

For one TP2 stage with per-GPU bytes \(B_g\):

\[
t_{\mathrm{H2D,stage}}=\max\!\left(
\frac{B_g}{22.0\ \mathrm{GB/s}},
\frac{2B_g}{25.4\ \mathrm{GB/s}},
\frac{2B_g}{180\ \mathrm{GB/s}}
\right).
\]

## No-ShadowKV native-cache control

Case C removes landmarks, low-rank key reconstruction, sparse-selection overlays,
host KV fetches, and whole-layer offload. It does not erase sparsity or compression
that is native to a checkpoint. Kimi therefore scans dense MLA context; GLM and GLM
Flash retain DSA; GLM Flash also retains 34 fixed-state KDA layers; and V4 Flash
retains its 128-token windows and official 4×/128× compressed streams.

For stage \(p\), native main-cache entries \(n_l(N)\), optional index-cache entries
\(i_l(N)\), cached widths \(d_c,d_i\), storage bytes/value \(q\), and TP degree \(G_s\):

\[
K_{\mathrm{GPU}}(N,q)=
\max_p\frac{q}{G_s}\sum_{l\in p}[n_l(N)d_c+i_l(N)d_i].
\]

The no-offload admission ceiling is memory-only:

\[
H_{\mathrm{GPU}}=0.90(80\ \mathrm{GiB})-W_{\mathrm{GPU}}-6\ \mathrm{GiB},
\qquad
C_{\max}=R\left\lfloor\frac{H_{\mathrm{GPU}}}{K_{\mathrm{GPU}}}\right\rfloor.
\]

The first request above \(C_{\max}\) is rejected; no host spill or hidden wait queue is
modeled. The index allocation uses one 128-value key per indexed position shared
across heads, matching the V4 Flash reference tensor. GLM uses the positions marked
`full` in its official `indexer_types`; GLM Flash pools index positions by four.

Native index scoring and main attention are sequential. Each receives a roofline over
BF16 math, measured effective HBM bandwidth, and fused FP8 conversion:

\[
t_x=\max\!\left(
\frac{F_x}{\eta_F\,F_{\max}},
\frac{B_x}{\eta_B\,B_{\mathrm{HBM}}},
\frac{V_x}{D_{\mathrm{FP8}}}
\right),
\qquad t_{\mathrm{attn},p}=t_{\mathrm{index},p}+t_{\mathrm{main},p}.
\]

This is an optimistic analytical native-attention bound, not a measured sparse-gather
kernel. It assumes ideal cache sharding across each TP group.

## TPOT and throughput

For a PP microbatch of \(b\) sequences, let \(s_p(b)\) be the generated core
time of stage \(p\), its local ShadowKV critical work, and its outgoing PP edge.
With \(g=\lceil C/b\rceil\) request groups, the LLMServingSim in-flight cap gives:

\[
T_{\mathrm{compute}}(C,b)=\max(P,g)\max_p s_p(b).
\]

Token-ahead copies overlap compute, but in steady state each group contributes one
node-local prefetch payload. Thus they also impose the service-rate constraint

\[
T_{\mathrm{H2D}}(C,b)=g\,t_{\mathrm{prefetch,node}}(b),
\qquad
T_{\mathrm{step}}(C,b)=\max(T_{\mathrm{compute}},T_{\mathrm{H2D}}).
\]

The known-request scheduler searches
\(1\le b\le\lceil C/P\rceil\). Larger microbatches would leave fewer than \(P\)
groups, underfill the pipeline, and cannot improve a monotone stage service curve.
PP=1 placements retain the complete continuous batch.

For Case C, \(s_p(b)\) replaces the ShadowKV critical work with native index and
attention roofline time. It has no H2D service constraint, and the search stops at the
memory-only admission ceiling above.

\[
Y_{\mathrm{total}}=\frac{1000C}{T_{\mathrm{step}}},
\qquad
Y_{\mathrm{user}}=\frac{1000}{T_{\mathrm{step}}},
\qquad
\mathrm{TPOT}=T_{\mathrm{step}}.
\]

DeepSeek V4 Flash divides users over two replicas before the profile lookup.

## Prefill and TTFT

Prefill uses 2K chunks and the same GenZ hardware. It includes active-path
projection/MoE work, official stored bytes per chunk, and architecture-specific
attention arithmetic: dense MLA for Kimi; DSA indexing for GLM; KDA plus pooled DSA
for GLM Flash; and official 4×/128× compression for V4 Flash.

For PP placements the fill multiplier is
\((n_{\mathrm{chunk}}+P-1)/n_{\mathrm{chunk}}\). Cold-burst TTFT uses:

\[
\overline{\mathrm{TTFT}}\approx T_0\frac{C_r+1}{2},
\qquad
\mathrm{TTFT}_{\mathrm{last}}\approx T_0C_r,
\qquad C_r=\left\lceil\frac{C}{R}\right\rceil.
\]

## MTP and whole-layer residency

For MTP-capable models, verification of \(k\) positions queries the generated core
profile at the enlarged batch:

\[
T_{\mathrm{core,verify}}(C,k)=T_{\mathrm{profile}}(kC).
\]

If the mean accepted prefix is \(A\), the round emits \(A+1\) tokens. In Cases A/B,
the oracle applies only to the first position and later positions pay fetch-at-decode
cache work. Case C instead verifies every position against the native HBM cache.

Residency requests are rounded to implementable whole layers:

\[
n_{\mathrm{resident}}=\operatorname{round}(pL),
\qquad p_{\mathrm{exact}}=n_{\mathrm{resident}}/L.
\]

Resident layers lose host fetch and low-rank reconstruction. The chart plots exact
ratios and interpolates only between adjacent implementable points.

## Sensitivity

The p10–p90 band is a deterministic parameter-sensitivity interval, not a confidence
interval. It varies compute efficiency, measured GEMM MBU, HBM, PCIe, host DRAM,
FP8/INT4 conversion, NVLink payload, selector timing, and materialization timing. The
no-ShadowKV interval varies the compute, HBM, conversion, and NVLink subset only.

The central microbatch choice is held fixed across sensitivity samples. This avoids
turning each uncertainty draw into a different scheduler and makes the band describe
hardware/kernel uncertainty around one explicit operating policy.

Important omissions remain: real fused frontier-model kernels, routing skew, expert
imbalance, framework launch gaps, A800-specific collectives, mixed prefill/decode
interference, and quality validation. Treat centers as planning values and expect at
least ±50% error until full operator profiles are available.
