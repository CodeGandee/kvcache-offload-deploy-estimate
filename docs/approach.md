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

## 5. Model the two cache-fetch cases

### Case A: token-ahead prediction

With predictor hit rate \(h=0.8\), false-positive prefetches and just-in-time misses
produce

\[
B_{\mathrm{total}}=(2-h)B_{\mathrm{sel}}=1.2B_{\mathrm{sel}},
\qquad
B_{\mathrm{JIT}}=(1-h)B_{\mathrm{sel}}=0.2B_{\mathrm{sel}}.
\]

The 80% correct transfer is assumed to overlap useful work; the 20% miss is exposed.

### Case B: fetch at decode

Landmark selection and miss discovery occur on the current query. With temporal
reuse \(r=0.60\), the current step materializes

\[
S_{\mathrm{miss}}=(1-r)S=0.4S.
\]

Key and value branches may overlap with each other, but sparse attention waits for
both. Per layer,

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

relative to BF16 KV. Because A800/SM80 lacks Hopper's native FP8 conversion path,
the estimate uses a fused load/dequantize/attention path with a more conservative
effective cache-path ratio \(q_{\mathrm{eff}}=0.72\). Attention still computes in
FP16/BF16.

## 8. Separate prefill TTFT from decode TPOT

Long prompts are split into 4K-token chunks. A 128K prompt supplies roughly 32
pipeline microbatches and a 256K prompt supplies 64, giving PP8 fill efficiencies
of \(32/39\) and \(64/71\). This makes single-prompt TTFT only moderately slower
than PP2, while single-user autoregressive decode is much slower.

For a simultaneous cold burst scheduled first-token-first,

\[
\overline{\mathrm{TTFT}}\approx T_0\frac{C_r+1}{2},
\qquad
\mathrm{TTFT}_{\mathrm{last}}\approx T_0C_r,
\qquad C_r=\left\lceil\frac{C}{R}\right\rceil.
\]

At sustained offered utilization \(\rho\to1\), queue delay is unbounded. The report's
100% point is a finite closed batch, not a stable production operating target.

## 9. Extend the repository with another case

A new deployment case should add:

1. A Markdown case summary in `docs/cases/`.
2. Its interactive or static artifact beside the summary.
3. Hardware, checkpoint, topology, cache, scheduler, and uncertainty assumptions.
4. Formula changes in the Python package when the analytical model changes.
5. Unit tests for new equations and an integration test for the published artifact.
6. Pinned external implementations under `extern/tracked/` when new code is used.

Do not silently reuse A800 bandwidth, PP8 fill, or ShadowKV selector constants for a
different server or algorithm.
