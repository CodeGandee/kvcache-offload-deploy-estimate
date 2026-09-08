# Estimation approach

This repository uses a reproducible three-part pipeline:

1. official Hugging Face configs define each model's dimensions and attention type;
2. GenZ converts operator work and the measured A100 hardware envelope into
   decode/prefill roofline timings;
3. generated timings are written in LLMServingSim's profile-bundle format and read
   through LLMServingSim's production interpolation path before external ShadowKV
   events, native attention for layers outside the overlay, and the KDA state-update
   roofline are added. The no-ShadowKV control uses the same native-attention and KDA
   extensions.

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

The cache placement is **pure TP with decode-context parallel degree 1**. Pipeline
stages own disjoint layer ranges, but an MLA latent row and a shared sparse-index row
are not head-sharded: each TP rank in the stage retains a full copy. Query-head
arithmetic remains TP-sharded. GLM-Flash's KDA recurrent state is also head-sharded.
This distinction replaces the earlier blanket `1/TP` cache factor.

The compact 576/512-value MLA cache and sparse-attention byte model assume an optimized
custom MLA serving kernel that consumes the shared latent directly.
In particular, Kimi's checked-in Transformers reference expands its latent state into
per-head K/V before updating the standard cache; running that literal cache path would
use materially more memory and attention bandwidth than this report. The direct-head
arithmetic roof used below is therefore an optimistic lower-bound adapter, not a
measured matrix-absorbed MLA kernel.

As a geometry cross-check before deployment placement, the complete BF16 persistent
state of one 256K sequence is 17.15625 GiB for Kimi, 22.61426 GiB for GLM
(21.9375 GiB MLA plus 0.67676 GiB index), and 2.97610 GiB for GLM Flash
(2.75 GiB MLA, 0.08862 GiB pooled index, and 0.13748 GiB fixed KDA). PP then assigns
subsets of those layers to stages. Pure TP replicates each stage's MLA/index portion;
only the GLM-Flash KDA portion is divided by TP8. These logical totals are explicit
regression tests, so a future blanket `/TP` change will fail validation.

## LLMServingSim bridge

The GenZ sweep covers 1–4,096 simultaneous sequences and is written to:

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

For context \(N\), selected fraction \(\alpha=1/64\), cache-bearing layers \(L_p\)
on pipeline stage \(p\), cached width \(d_c\), and stored bytes/value \(b\):

\[
S(N)=\lceil\alpha N\rceil,
\qquad
B_{\mathrm{sel,GPU},p}=S(N)L_pd_cb.
\]

The selected latent rows are replicated under pure TP. The direct-copy baseline has
every TP rank DMA its own copy, so a stage with TP degree \(G_s\) places
\(G_sB_{\mathrm{sel,GPU},p}\) on the node's host-memory service path. A runtime that
loads once and broadcasts over NVLink is a separate optimization and is not assumed.

After 60% temporal reuse, the central 80%-recall/80%-precision token-ahead oracle
prefetches 0.40 of the selected set and leaves 0.08 just in time:

\[
f_{\mathrm{prefetch}}=(1-r)\frac{R_o}{P_o}=0.40,
\qquad
f_{\mathrm{JIT}}=(1-r)(1-R_o)=0.08.
\]

Here temporal reuse is one-token working-set overlap: 60% of the entries selected
for the current token are assumed to remain in the GPU sparse buffer from the prior
token. Recall and precision are then evaluated only over the remaining 40% true
miss set. For a 100-entry selected set, 60 are resident; the predictor correctly
prefetches 32 of the 40 misses. At 80% precision it prefetches 40 entries in total,
so 8 are false positives, while the 8 false negatives are fetched just in time.
Total transfer is therefore 48 entry-equivalents, but only 8 remain on the ideal
critical path.

The 60% central value is adapted from ShadowKV's reported roughly 60% chunk hit
rate. The 80% oracle recall and precision are hypothetical sensitivity assumptions
for the proposed token-ahead extension, not measurements reported by ShadowKV or
validated on the target frontier models.

Prefetch overlaps the preceding token's model-forward window. The current token
still verifies landmarks. Fetch-at-decode has no background prefetch and exposes
the full 40% post-reuse miss set.

For a cache-bearing block:

\[
t_{\ell}=t_{\mathrm{select},\ell}
+\max(t_{K,\ell},t_{V,\ell})
+t_{\mathrm{attention},\ell}.
\]

ShadowKV Table 13 grounds three separate centers: landmark selection, overlapped
materialization, and sparse attention. At 24×128K those are
(0.58+0.07+0.15=0.80) ms, (max(1.36,1.66)=1.66) ms, and 0.21 ms per Llama
block. The attention term is not hidden inside materialization: it is scaled by the
selected-row count and the custom joint-latent width. Its lower bound is

\[
t_{\mathrm{attention},\ell}=\max\!\left(
t_{\mathrm{Table13,scaled}},
\frac{F_{\mathrm{sparse}}}{\eta_F F_{\max}},
\mathbf 1_{\mathrm{FP8}}\frac{V_{\mathrm{selected}}}{D_{\mathrm{FP8}}}
\right).
\]

Thus FP8 conversion is charged exactly once, when attention consumes the complete
selected set, including temporally reused or full-resident rows. It is not added to
miss materialization. The Table 13 memory center is preserved on the central A100
proxy and scaled with measured HBM bandwidth; no separate uncited launch floor or
GEMM-MBU substitution is added. Reusing these Llama per-block measurements on the
target architectures remains a major extrapolation.

For GLM Flash, GenZ includes projection/weight work but not the recurrent KDA state
update. If stage \(p\) has \(L_{K,p}\) KDA layers, logical fixed-state bytes \(F_p\),
TP degree \(G_s\), local sequence count \(b\), and \(k\) jointly verified positions,
the separate lower bound is

\[
t_{\mathrm{KDA},p}=\max\!\left(
\frac{7bkL_{K,p}(64/G_s)128^2}{\eta_F\,F_{\max}},
\frac{2b(F_p/G_s)}{\eta_B\,B_{\mathrm{HBM}}}
\right).
\]

The factor seven represents decay, state-key contraction, outer-product update, and
state-query contraction. A fused multi-token verification reads and writes persistent
state once per sequence while arithmetic scales with \(k\). This is a hardware lower
bound rather than a measured fused KDA kernel; mixed FP32/BF16 arithmetic and launches
can make a real implementation slower.

For one TP2 stage with per-GPU bytes \(B_g\):

\[
t_{\mathrm{H2D,stage}}=\max\!\left(
\frac{B_g}{22.0\ \mathrm{GB/s}},
\frac{2B_g}{25.4\ \mathrm{GB/s}},
\frac{2B_g}{180\ \mathrm{GB/s}}
\right).
\]

## ShadowKV HBM admission

Cases A and B use the same memory-only admission definition as the native-cache
control. System-RAM capacity is unbounded: the exact offloaded cache can grow without
causing admission failure. Host-DRAM and H2D bandwidth are still finite and remain in
the TPOT model.

The HBM footprint follows the public ShadowKV representation instead of equating the
selected fraction with stored state. For context length \(N\), rank \(r_k=160\),
chunk size \(c=8\), selected count \(S=\lceil N/64\rceil\), outlier chunks
\(O=24\lfloor S/1024\rfloor\), cached width \(d_c\), landmark count \(n_L\), and
sparse-buffer count \(n_B\):

\[
n_L=\max\!\left(0,\left\lfloor\frac{N}{c}\right\rfloor-4-O\right),
\qquad
n_B=S+128+c(O+4),
\]

\[
V_{\mathrm{shadow,layer}}=Nr_k+r_kd_c+n_Ld_c+n_Bd_c.
\]

These terms represent the low-rank \(U\), low-rank \(SV\), chunk landmarks, and
selected/outlier/local exact buffer. A fully resident overlaid layer uses
\(Nd_c+n_Ld_c\). Native state outside the overlay remains in HBM in its official
compressed or recurrent form.

The number of layers is part of the per-GPU footprint. For main-cache bytes/value
\(q\), ShadowKV layers \(\mathcal L_{S,p}\), native non-overlaid layers
\(\mathcal L_{N,p}\), retained native index states \(\mathcal I_p\), index bytes per
entry \(e_i\), and TP-sharded fixed KDA bytes \(F_p/G_s\) on PP stage \(p\):

\[
K_{\mathrm{GPU}}(q)=\max_p\!\left\{
q\!\left[
\sum_{\ell\in\mathcal L_{S,p}}V_{\mathrm{shadow},\ell}
+\sum_{\ell\in\mathcal L_{N,p}}V_{\mathrm{native},\ell}
\right]
+\sum_{i\in\mathcal I_p}n_i e_i
+\frac{F_p}{G_s}
\right\}.
\]

GLM and GLM-Flash index entries remain 132 bytes in both main-cache modes: 128 FP8
values plus one FP32 scale. The checked-in V4 Flash reference keeps its 128-value
index row in BF16, or 256 bytes. The ShadowKV cases conservatively retain these
trained index states in addition to the low-rank/landmark overlay; ShadowKV's measured
selector timing is used as the selection proxy rather than charging a second index
kernel. GLM-Flash also retains 34 fixed KDA states; each layer contributes a 4 MiB
FP32 recurrent matrix plus about 0.140625 MiB of BF16 convolution state before TP8
head sharding.

For Kimi 128K FP8, the busiest PP8 stage has eight layers. Native cache is
\(8\times131{,}072\times576\times1.03125/2^{30}\approx0.5801\)
GiB/request/GPU, giving 54 users from 31.35 GiB of headroom. The modeled ShadowKV
representation is 0.2456 GiB/request/GPU, giving 127 users. There is no `/TP2`
factor in either pure-TP value.

For the maximum-loaded stage GPU, the common planning OOM boundary is:

\[
H_{\mathrm{GPU}}=0.90(80\ \mathrm{GiB})-W_{\mathrm{GPU}}-6\ \mathrm{GiB},
\qquad
C_{\max}=R\left\lfloor\frac{H_{\mathrm{GPU}}}{K_{\mathrm{GPU}}}\right\rfloor.
\]

Here (W_{\mathrm{GPU}}) is the checkpoint's total stored bytes divided evenly over
the GPUs that hold one replica. It is not a stage-specific packed-weight footprint.
Cache state is still evaluated on the busiest PP stage, so uneven real weight packing
can move the Kimi/GLM whole-request boundary by a few requests. Exact deployment
admission should replace this average with the final per-stage allocation manifest.

The next whole request is rejected. The 90% allocation fraction makes this a
configured OOM boundary rather than a claim that every physical HBM byte is usable.

## No-ShadowKV native-cache control

Case C removes landmarks, low-rank key reconstruction, sparse-selection overlays,
host KV fetches, and whole-layer offload. It does not erase sparsity or compression
that is native to a checkpoint. Kimi therefore scans dense MLA context; GLM and GLM
Flash retain DSA; GLM Flash also retains 34 fixed-state KDA layers; and V4 Flash
retains its 128-token windows and official 4×/128× compressed streams.

For stage \(p\), native main-cache entries \(n_l(N)\), index-cache entries
\(i_l(N)\), main cached width \(d_c\), main-cache bytes/value \(q\), fixed index-row
bytes \(e_i\), and fixed KDA state \(F_p\):

\[
K_{\mathrm{GPU}}(N,q)=
\max_p\left[
q\sum_{l\in p}n_l(N)d_c
+\sum_{l\in p}i_l(N)e_i
+\frac{F_p}{G_s}
\right].
\]

The no-offload admission ceiling uses the same HBM-only rule:

\[
H_{\mathrm{GPU}}=0.90(80\ \mathrm{GiB})-W_{\mathrm{GPU}}-6\ \mathrm{GiB},
\qquad
C_{\max}=R\left\lfloor\frac{H_{\mathrm{GPU}}}{K_{\mathrm{GPU}}}\right\rfloor.
\]

The first request above \(C_{\max}\) is rejected; no host spill or hidden wait queue is
modeled. System-RAM capacity is likewise irrelevant to this case. The index allocation
uses one 128-value key per indexed position shared across heads. GLM uses the positions
marked `full` in its official `indexer_types`; GLM Flash pools index positions by four.
Its index format is independent of the BF16/FP8 main-cache toggle.

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
kernel. HBM bytes and cache conversion are replicated per TP rank, while query-head
FLOPs retain the \(1/G_s\) tensor-parallel factor.

GLM Flash additionally pays \(t_{\mathrm{KDA},p}\) above in both ShadowKV and native
cases. In the V4 ShadowKV cases, ratio-128/local-window layers outside the ratio-4
overlay still pay their official native-attention roofline; they are not absorbed into
the GenZ core or the sparse ShadowKV term.

Decode context parallelism could sequence-shard the growing history by an explicit
degree \(D\), but it would add distributed selection/top-k and partial-attention
communication. That is not modeled here, and the retired ideal-sharding curves must
not be relabeled as DCP results.

## TPOT and throughput

For a PP microbatch of \(b\) sequences, let \(s_p(b)\) be the generated core
time of stage \(p\), its local ShadowKV critical work, native attention for any
non-overlaid layer, KDA state-update roof, and its outgoing PP edge.
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

For Case C, \(s_p(b)\) replaces the ShadowKV and non-overlaid terms with native index
and attention roofline time while retaining the KDA update roof. It has no H2D service
constraint, and the search stops at the memory-only admission ceiling above.

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
cache work. Every target position still pays sparse attention. Case C instead verifies
every position against the native HBM cache. For GLM Flash, fused target verification
scales KDA arithmetic by \(k\) but reads/writes persistent state once; the approximate
one-layer draft charge includes the average KDA-layer share.

Residency requests are rounded to implementable whole layers:

\[
n_{\mathrm{resident}}=\operatorname{round}(pL),
\qquad p_{\mathrm{exact}}=n_{\mathrm{resident}}/L.
\]

Resident layers lose host fetch and low-rank reconstruction but retain landmark
selection and sparse attention:

\[
T_{\mathrm{shadow}}(n)=L(T_{\mathrm{select}}+T_{\mathrm{attention}})
+(L-n)T_{\mathrm{offload}}.
\]

FP8 conversion for the selected set is already inside \(T_{\mathrm{attention}}\), so
there is no extra resident-layer conversion term. The chart plots exact ratios and
interpolates only between adjacent implementable points.

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
