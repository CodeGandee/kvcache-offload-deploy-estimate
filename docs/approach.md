# Estimation approach

## 1. Separate measured inputs from modeled behavior

Checkpoint byte totals come from the official Hugging Face repositories. Hardware
capacities and link rates come from vendor documentation. ShadowKV selector timing
and reconstruction/fetch ordering come from the paper and its implementation.

Everything else—kernel fusion efficiency, pipeline balance, overlap, and final
throughput—is explicitly treated as a model assumption rather than a measurement.

## 2. Place official checkpoint formats

The current case retains each official storage format and converts fragments as they
are consumed for BF16 compute. It does not keep a second persistent BF16 model copy.

| Candidate | Official checkpoint bytes | Placement |
|---|---:|---|
| Kimi Code 2.7 | 595.2 GB native INT4 | PP8×TP2, four stages per server |
| GLM-5.3 | 755.6 GB FP8 plus metadata | PP8×TP2, four stages per server |
| DeepSeek V4 Flash | 159.6 GB mixed FP4/FP8 | two four-GPU replicas, one server |

For Kimi and GLM, the approximate balanced stage footprints are 74.4 GB and
94.5 GB respectively, or 37.2 GB and 47.2 GB per GPU before runtime buffers.

## 3. Keep exact cache stage-local

Stages 1–4 are placed on server A and stages 5–8 on server B. Each server stores the
exact KV data for its own half of the layers in NUMA-local host memory. The selected
entries move only from that server's RAM to its local GPUs; KV payload does not cross
InfiniBand.

PP8×TP2 and PP2×TP8 have the same balanced per-GPU weight and cache share:

\[
\frac{W}{8\cdot2}=\frac{W}{2\cdot8}=\frac{W}{16},
\qquad
\frac{L/8}{2}=\frac{L/2}{8}=\frac{L}{16}.
\]

Only the stage-4→5 BF16 activation crosses the 400 Gb/s InfiniBand link. Six other
pipeline boundaries remain inside the two servers on NVLink.

## 4. Compute the sparse working set

For context length \(N\) and selected fraction \(\alpha=1/64\),

\[
S(N)=\alpha N,
\qquad S(72\mathrm{K})=1{,}152,
\qquad S(128\mathrm{K})=2{,}048,
\qquad S(256\mathrm{K})=4{,}096.
\]

For \(L\) cache-bearing layers, exact cached width \(d_c\), value size \(b\),
\(P\) pipeline stages, and \(G_s\) GPUs per stage, the balanced selected payload per
GPU and request is

\[
B_{\mathrm{sel,GPU}}=
\frac{S(N)(L/P)d_cb}{G_s}.
\]

The Python implementation is in `src/kvcache_offload_deploy_estimate/model.py`.

## 5. Model the two cache-fetch cases with LLMServingSim events

The tracked LLMServingSim checkout supplies the vLLM-compatible transformer-block
pipeline partitioner and the vocabulary for per-stage traces. The ShadowKV extension
lives in `src/kvcache_offload_deploy_estimate/llmservingsim_shadowkv.py`; upstream is
not patched because its normal tiered block recall does not contain ShadowKV's
per-block landmark-selection dependency.

The extension inserts, for every cache-bearing transformer block:

1. landmark GEMM/softmax, max reduction, and top-K selection;
2. optional one-token-ahead background prefetch;
3. the current token's just-in-time miss materialization;
4. overlapping key reconstruction and value/latent fetch;
5. FP8 dequantization before BF16 attention compute.

ShadowKV Table 13 is a **per-transformer-block** measurement. Both selection and
`max(reconstruct K, fetch V)` are therefore charged once per cache-bearing block.

### Case A: token-ahead prediction

Let \(r=0.60\) be entries reused from the preceding token. Over the remaining
entries, the advisory oracle has recall \(R_o=0.80\) and precision \(P_o=0.80\).
Prefetch and just-in-time traffic are

\[
B_{\mathrm{prefetch}}=(1-r)\frac{R_o}{P_o}B_{\mathrm{sel}}=0.40B_{\mathrm{sel}},
\qquad
B_{\mathrm{JIT}}=(1-r)(1-R_o)B_{\mathrm{sel}}=0.08B_{\mathrm{sel}}.
\]

The current query still runs landmark selection to verify the exact set and discover
misses. Prefetch may overlap the preceding token; selection and the 0.08 miss set are
exposed. A perfect authoritative oracle would instead use recall=precision=1 and
could optionally skip verification, but that is not the central case.

### Case B: fetch at decode

Landmark selection and miss discovery occur on the current query. With temporal
reuse \(r=0.60\), the current step materializes

\[
S_{\mathrm{miss}}=(1-r)S=0.4S.
\]

Key and value branches may overlap with each other, but sparse attention waits for
both. Per layer, and then summed over every cache-bearing block,

\[
t_{\mathrm{layer}}=t_{\mathrm{QKV}}+t_{\mathrm{select}}
+\max(t_K,t_V)+t_{\mathrm{attention}}+t_{\mathrm{FFN}}.
\]

## 6. Model PP8×TP2 pipeline fill

Let \(m\) be independently schedulable decode microbatches, \(P=8\), and
\(t_*=\max_j t_j\) the slowest balanced-stage time. A forward wave takes

\[
T_{\mathrm{wave}}\approx(m+P-1)t_*=(m+7)t_*,
\qquad
\eta_{\mathrm{PP8}}=\frac{m}{m+7}.
\]

A PP8 stage has one eighth of the layers but one quarter of the tensor parallelism
of a TP8 stage, so its time is approximately half the one-node TP8 full-model time.
The ideal two-node throughput multiplier is therefore \(2m/(m+7)\), not
\(8m/(m+7)\).

Relative to the earlier PP2×TP8 estimate, the central ratio is

\[
\frac{Y_{\mathrm{PP8}}}{Y_{\mathrm{PP2}}}
\approx g_{\mathrm{TP2}}\frac{m+1}{m+7},
\qquad g_{\mathrm{TP2}}=1.03.
\]

The one-user ratio is set to 0.24 to include six extra intra-node stage handoffs.
The favorable central curve uses one request per decode microbatch, \(m=C\). If a
runtime groups \(g\) requests per microbatch, use \(m=\lceil C/g\rceil\), which
increases bubbles. Successive tokens from one user cannot fill the pipeline because
token \(t+1\) depends on token \(t\).

The eight stages must be balanced by measured time, not merely by layer count. The
slowest stage sets pipeline cadence, especially for heterogeneous MoE blocks.

## 7. Model FP8 KV storage on A800

One FP8 byte plus one FP32 scale per 128 values has byte ratio

\[
q_{\mathrm{bytes}}=\frac{1+4/128}{2}=0.515625
\]

relative to BF16 KV. The extension does not apply a blanket effective multiplier.
It scales stored-byte transfer by 0.515625, adds explicit dequantization at a central
360 Gvalue/s, and leaves low-rank reconstruction plus attention in BF16. The
sensitivity run samples 240–520 Gvalue/s. This makes FP8 gains small whenever key
reconstruction is the critical materialization branch.

## 8. Parameter sensitivity, not confidence

Each displayed center has a reproducible 256-sample sensitivity study. It varies
usable PCIe bandwidth (18–29 GB/s), node host-DRAM bandwidth (130–220 GB/s), FP8
dequantization (240–520 Gvalue/s), selector timing (20% log-normal spread), and
materialization timing (25% spread). The resulting p10–p90 range is not a confidence
interval: it omits architecture mismatch and unknown A800 frontier-model profiles.

## 9. Separate prefill TTFT from decode TPOT

Long prompts are split into 4K-token chunks. A 72K, 128K, or 256K prompt supplies
roughly 18, 32, or 64 pipeline microbatches, giving PP8 fill efficiencies of
\(18/25\), \(32/39\), and \(64/71\). This makes single-prompt TTFT only moderately
slower than PP2, while single-user autoregressive decode is much slower.

For a simultaneous cold burst scheduled first-token-first,

\[
\overline{\mathrm{TTFT}}\approx T_0\frac{C_r+1}{2},
\qquad
\mathrm{TTFT}_{\mathrm{last}}\approx T_0C_r,
\qquad C_r=\left\lceil\frac{C}{R}\right\rceil.
\]

At sustained offered utilization \(\rho\to1\), queue delay is unbounded. The report's
100% point is a finite closed batch, not a stable production operating target.

## 10. Add native MTP as a separate decode overlay

The model-fixed NextN layer count is not used as the speculative block length. The
serving overlay uses the documented deployment starting points: \(k=3\) draft tokens
for GLM-5.3, \(k=2\) for DeepSeek V4 Flash, and \(k=0\) for Kimi Code 2.7. The report
evaluates mean consecutively accepted prefixes \(A\in\{1,2\}\). A correct token after
an earlier rejection is not counted as accepted.

One speculative verification round emits \(A+1\) output tokens, including the target
correction or bonus token:

\[
\mathrm{TPOT}_{\mathrm{MTP}}(A)=
\frac{T_{\mathrm{verify}}(k)+T_{\mathrm{draft}}(k)}{A+1}.
\]

The calibrated non-ShadowKV target path is split into shared work and a fraction
\(\beta(C)\) that scales per additional verified position:

\[
T_{\mathrm{core,verify}}=T_{\mathrm{core}}[1+\beta(C)(k-1)].
\]

For PP8 models, \(\beta\) rises from 0.15 to 0.25 over the admitted load range; for
the single-stage V4 Flash replica it rises from 0.30 to 0.55. The remaining core path
represents shared weight reads, pipeline launch/bubble time, and communication. MTP
drafting is charged as \(1.5k/L\) of the core floor plus one sequential
selector/materialization block per draft step.

All \(k\) candidate positions pay target verification. The one-token-ahead KV oracle
covers only the first target position; deeper positions pay fetch-at-decode selection
and miss work. The separate MTP plots therefore do not imply a generic
\((A+1)\)-times speedup.

## 11. Quantize partial HBM residency to whole layers

For a requested exact-KV HBM ratio \(p\), the implementation can retain only an
integer number of cache-bearing layers:

\[
n_{\mathrm{resident}}=\operatorname{round}(pL),
\qquad
p_{\mathrm{exact}}=\frac{n_{\mathrm{resident}}}{L}.
\]

The scan evaluates requested ratios 0%, 10%, ..., 100%, distributes the resident
layers proportionally across PP stages, and plots \(p_{\mathrm{exact}}\). Markers are
modeled placements. Values between adjacent markers use linear interpolation:

\[
y(p)=y_i+\frac{p-p_i}{p_{i+1}-p_i}(y_{i+1}-y_i).
\]

A resident layer keeps full exact K/V in HBM. Landmark/low-rank scoring still selects
the 1.56% attention working set, but host fetch and low-rank key reconstruction are
removed for that layer. FP8-resident KV still pays conversion before BF16 compute.
The per-stage critical addition becomes

\[
T_{\mathrm{shadow}}(n)=LT_{\mathrm{select}}
+(L-n)T_{\mathrm{offload}}
+nT_{\mathrm{HBM,dequant}}.
\]

The maximum resident footprint among stage GPUs is checked against
\(0.90\times80\) GiB minus stored checkpoint weights and a 6 GiB/GPU runtime reserve.
Hollow chart markers exceed this planning envelope. This check is optimistic because
mandatory low-rank bases, landmarks, CUDA graphs, and fragmentation are not separately
sized.

The residency explorer has a discrete load slider for one user and 10%, 20%, ...,
100% of each scenario's admission ceiling. Its curves use the no-MTP target path so
the residency effect is not conflated with speculative acceptance.

## 12. Add the 72K context point

The 72K point is 73,728 tokens. Its non-ShadowKV decode floors are calibrated
extrapolations rather than measurements. Admission ceilings are 48 users for Kimi,
48 for GLM, and 80 across the two V4 Flash replicas; single-prompt TTFT centers are
7.5, 8.0, and 4.1 seconds. These values have at least the same ±50% architecture and
kernel uncertainty as the 128K/256K points.

## 13. Extend the repository with another case

A new deployment case should add:

1. A Markdown case summary in `docs/cases/`.
2. Its interactive or static artifact beside the summary.
3. Hardware, checkpoint, topology, cache, scheduler, and uncertainty assumptions.
4. Formula changes in the Python package when the analytical model changes.
5. Unit tests for new equations and an integration test for the published artifact.
6. Pinned external implementations under `extern/tracked/` when new code is used.

Do not silently reuse A800 bandwidth, PP8 fill, or ShadowKV selector constants for a
different server or algorithm.
