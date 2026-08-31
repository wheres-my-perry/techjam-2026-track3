# Perry — Accuracy-Gated Transformer Optimization on RTX 5090

**Perry is a standalone, AI-assisted PyTorch Transformer implementation that
passes the strict correctness gate on all 14 official TikTok TechJam 2026 Track
3 shapes. On an NVIDIA GeForce RTX 5090, it achieves an 11.803x geometric-mean
speedup on shapes #1–#13 and makes the extreme `B=32, S=100000, D=1024` shape
#14 executable within 32 GiB of GPU memory.**

## The problem

The organizer's reference Transformer is intentionally easy to inspect rather
than optimized for GPU execution. Each block launches three separate Q/K/V
projections, explicitly forms the attention-score matrix, applies masking and
FP32 softmax, multiplies by V, and then runs an output projection and a
two-layer GELU feed-forward network. This creates repeated input reads, many
kernel launches, expensive intermediate tensors, and avoidable memory traffic.

The 14 official shapes also represent very different performance regimes. A
small batch may be dominated by Python and launch overhead; `B=10000` produces
large projection and FFN workloads despite a short sequence; `D=1024` stresses
Tensor Core throughput; and `S=1024` makes attention layout increasingly
important. Shape #14 is qualitatively different: its explicit reference
attention score would require approximately 18.6 TiB, so it is a memory
feasibility problem as well as a compute problem.

Performance only counts when every output element satisfies the competition's
strict logical OR: `absolute_error < 0.002 OR relative_error < 0.02`. We kept
those strict `<` comparisons unchanged. A candidate that missed by even one
element was rejected before performance measurement.

## Our core insight

There is no single universally optimal Transformer kernel across these shapes.
The reference pays three different kinds of cost: arithmetic from explicit
attention, bandwidth and launch cost from moving intermediate tensors, and
peak-memory cost from keeping too many tensors live at once. Which cost
dominates changes with `B`, `S`, `D`, and the number of heads. Optimizing only
one kernel therefore cannot solve the complete benchmark.

Our strategy was to optimize at three levels while preserving one public
interface, `forward(x, valid_token_mask)`:

1. Reduce graph, data-movement, and launch overhead using packed projections,
   Flash-first SDPA, custom fusion, and whole-model compilation.
2. Use lower precision only where Tensor Core throughput helps, while keeping
   numerically sensitive boundaries in FP32.
3. Change execution scheduling when tensor lifetime—not the Transformer
   formula—is the reason a valid workload cannot fit in memory.

The development loop was always the same: establish the unchanged reference as
the oracle, profile a real official shape, make one targeted change, run the
strict accuracy gate, and then compare end-to-end latency on the same GPU. A
faster microkernel was useful only if it improved the complete model without
spending the numerical margin needed by another layer.

## From the reference to the final architecture

The table below is cumulative: every `+` stage includes the optimizations above
it. All speedups come from one chronological sweep on the same RTX 5090 host;
the `1.001x` reference row shows ordinary measurement noise around `1.0x`.

| Cumulative optimization stage | Geometric-mean speedup | Main lesson |
|---|---:|---|
| Reference PyTorch Transformer | 1.001x | Control measurement |
| + Packed QKV Projection | 1.076x | Reusing one input read helps, but projection packing alone is modest |
| + Scaled Dot-Product Attention | 1.435x | Fused attention removes expensive score-path work |
| + No-Copy QKV and Causal-Mask Elision | 2.101x | `+46.35%` over the preceding stage by reducing views, data movement, and causal-mask overhead |
| + Mixed-Precision Tensor Core Path and Compilation | 10.200x | The largest step combines two changes, so we do not assign it to either one alone |
| + Shape-Aware SDPA Backend Dispatch | 10.449x | A small `+2.44%` gain that remained positive in reverse order |
| + Flash-First Causal Attention | 11.676x | A stable `+11.74%` gain over backend dispatch |
| + Triton FFN-In and Exact-GELU Fusion | 11.785x | Useful fusion, but the sub-1% aggregate difference was order-sensitive |
| + FP32 Pre-GELU Accumulation | 11.748x | Recovered numerical margin while staying in the same performance tier |
| + Memory-Bounded Long-Sequence Scheduling — Final | **11.803x** | Added shape-#14 execution and a clean standalone submission |

For the other stages, worst maximum absolute error was `0.000607941` after
packed QKV, `0.00132750` after SDPA, `0.00188218` after mixed precision and
backend dispatch, and approximately `0.00188214` after Flash-first attention
and Triton fusion. FP32 pre-GELU accumulation reduced it to `0.00179085`, which
the final architecture retained. These summaries never replace the full gate.

Reverse-order controls place Flash-first attention, Triton fusion, FP32
pre-GELU accumulation, and the final submission in the same `11.7–11.8x`
performance band. We selected the final architecture for numerical margin,
standalone integration, and executable shape #14—not a marginal aggregate lead.
Shape-level results reinforce the need for specialization: shape #13 reaches
`38.762x`, while wide-model shape #8 reaches `2.379x`.

## How we used AI

[OpenAI Codex](https://developers.openai.com/) was our development-time coding
agent and engineering collaborator. It helped audit the statement and strict
comparator, map the 14 shapes into bottleneck regimes, propose experiments,
implement and review PyTorch/Triton candidates, build isolated benchmark
runners, and construct the memory-bounded shape-#14 validator.

Codex helped interpret profiler traces and raw GPU events; shape-#14 profiling
showed attention consumed 92.258% of the compiled executor's device time. It
also maintained architecture, experiment, and decision logs.

AI suggestions were never evidence: every promoted change passed the unchanged
gate and target-GPU measurement. The submitted runtime calls no OpenAI API or
other external application service.

## Architecture

The submitted path is
[`main.py`](https://github.com/wheres-my-perry/techjam-2026-track3/blob/main/main.py)
→
[`v16_1_clean.py`](https://github.com/wheres-my-perry/techjam-2026-track3/blob/main/v16_1_clean.py).
The latter is a standalone model/config/kernel artifact: it imports PyTorch and
optional Triton, but no benchmark harness or historical implementation.

> **Input `[B,S,D]` + valid-token mask**<br>
> ├─ `S < 8192`: whole-model `torch.compile(max-autotune)` path<br>
> └─ `S >= 8192`, `B > 1`: eager batch loop → reusable compiled `B=1` body<br>
> &nbsp;&nbsp;&nbsp;↓ each Transformer block<br>
> FP32 LayerNorm → FP16 packed QKV → Flash-first SDPA → FP16 projection → FP32 residual<br>
> FP32 LayerNorm → Triton FFN-in + exact GELU → FP16 FFN-out → FP32 residual<br>
> &nbsp;&nbsp;&nbsp;↓<br>
> FP32 final LayerNorm and output `[B,S,D]`

For shapes #1–#13, the benchmark compiles the complete model after weight loading,
device transfer, and `eval()`, exposing all four Transformer blocks to
TorchInductor. Shape #14 deliberately breaks that outer graph: compiling the
32-sample loop could unroll it or extend tensor lifetimes, so only its reusable
single-sample body is compiled.

The fast path targets FP32 inference with internal FP16 compute. Non-causal
cases retain their key mask and use automatic SDPA; training and unsupported
public dtypes execute the reference arithmetic. Flash is a priority rather than
an assumption: PyTorch can fall through to cuDNN, memory-efficient, or math
SDPA when required. Parameter names remain compatible with the reference
`state_dict`. Derived FP16 weights and the compiled long-sequence executor are
runtime caches, not model parameters; both are refreshed or invalidated after
weight loading, device/dtype moves, and relevant mode changes.

## Algorithm and implementation details

### 1. Packed QKV, no-copy head views, and whole-graph execution

**Bottleneck and insight.** The reference applies three independent linear
projections to the same normalized tensor. That means three GEMM dispatches,
three reads of the same activation, and three separately produced outputs per
layer. Packing the projection is safe because Q, K, and V have the same input
and output dimensions; concatenating their weights along the output dimension
does not change any dot product.

**Implementation.** Perry stores a derived packed weight
`[W_Q; W_K; W_V]` and bias in FP16 and performs one `F.linear`, producing
`[B,S,3D]`. Instead of splitting Q/K/V and calling `.contiguous()` three times,
it reshapes to `[B,S,3,H,Dh]`, permutes, and unbinds three views in the layout
accepted by SDPA. A flattened model loop then exposes packed projection,
attention, residuals, and FFN operations from every layer to one
`torch.compile(mode="max-autotune")` graph. Compilation and autotuning finish
during warmup and are excluded from steady-state timing.

**Correctness and lifecycle.** The original `q_proj`, `k_proj`, and `v_proj`
parameters remain present, so strict reference weight loading still works.
Packed FP16 tensors are non-persistent caches: they do not enter the public
`state_dict`, refresh after `load_state_dict()` and `.to()`, and are not used in
training, where optimizer updates could make them stale.

**Measured result and tradeoff.** Packed QKV alone moved the same-host
geometric mean only from `1.001x` to `1.076x`; it removed real work but did not
touch explicit attention or the rest of the launch chain. The cumulative
no-copy/mask-elision stage reached `2.101x`, including a `46.35%` improvement
over the preceding SDPA stage. The subsequent jump to `10.200x` combined
mixed-precision Tensor Core execution with compilation, so we intentionally do
not claim that either change alone produced the full increase.

### 2. Flash-first causal attention with a mask proof

**Bottleneck and insight.** The reference writes `QKᵀ`, applies masks,
materializes softmax probabilities, and multiplies them by V. Those
sequence-squared intermediates dominate long attention and make shape #14's
original score tensor impossible to allocate. PyTorch SDPA can execute the same
dense attention through a fused backend, but a non-null padding mask can prevent
the Flash path even when the mask carries no additional information.

For causal self-attention with right padding, valid tokens form a prefix. A
valid query at position `i` can only attend to keys at positions `j ≤ i`, all
of which are also inside that valid prefix. It can never observe a padded key in
the suffix. Perry therefore passes `attn_mask=None`, sets `is_causal=True`, and
zeros invalid query positions at the block boundary. Non-causal execution keeps
the key mask. This proof does not extend to arbitrary sparse or left-padding
masks, so those inputs are not silently treated as equivalent.

**Implementation and fallback.** The backend priority is Flash → cuDNN →
memory-efficient → math; unsupported choices fall through inside PyTorch
without a host-side tensor check or synchronization. The scale remains exactly
`1/sqrt(head_dim)`, dropout is zero, and the result follows the same output
projection and residual path as the reference.

**Measured result and tradeoff.** Replacing explicit attention with SDPA raised
the early cumulative result from `1.076x` to `1.435x`. Later, changing the
compiled path from shape-specific backend dispatch to Flash-first attention
raised the same-host geometric mean from `10.449x` to `11.676x`, a stable
`11.74%` increase that remained positive in reverse-order controls. The
tradeoff is a narrower mask contract on the optimized causal path; safe
fallbacks remain available outside it.

### 3. Mixed precision as protected precision islands

**Bottleneck and insight.** The RTX 5090 provides much higher throughput for
FP16 Tensor Core operands, but converting the entire Transformer to low
precision accumulated too much error across four residual blocks. The useful
unit of optimization was therefore not “the model dtype,” but a set of
carefully bounded precision islands.

The final arithmetic boundary is explicit:

- Public inputs and outputs, LayerNorm, residual streams, and final LayerNorm
  remain FP32.
- Packed QKV, SDPA, attention output projection, and both FFN GEMMs use FP16
  activations and cached FP16 weights.
- Projection results return to FP32 before residual addition.
- The custom FFN-in dot product accumulates in FP32; exact GELU reads the FP32
  accumulator plus bias, and only its output is stored in FP16 for FFN-out.

This boundary keeps high-throughput GEMMs and attention on Tensor Cores while
preventing low-precision residual drift from compounding across layers. The
mixed-precision/backend stages had a worst maximum absolute error of
`0.00188218`. Removing the pre-GELU rounding boundary reduced the final worst
maximum absolute error to `0.00179085` without adding a tensor or launch. More
aggressive BF16, FP8, INT8, or full-FP16-accumulation candidates were rejected
by the unchanged elementwise gate and are summarized later.

### 4. Triton FFN-in and exact-GELU fusion

**Bottleneck and insight.** After attention improved, profiling exposed the
first FFN projection and its activation boundary as another opportunity. A
separate FFN-in linear and exact GELU produce an intermediate and leave less
control over where FP16 rounding occurs. Fusing the complete two-layer MLP looked
attractive, but isolated kernel speed did not translate consistently to
whole-model latency.

**Implementation.** A Triton custom operator fuses FFN-in matrix
multiplication, bias, and exact erf-GELU. Its tiled dot product reads FP16
operands into an FP32 accumulator, applies the bias and GELU before the first
rounding boundary, and writes one FP16 hidden tensor for the existing FFN-out
GEMM. `torch.library.custom_op` plus a fake implementation lets TorchInductor
capture the operator; a mathematically equivalent PyTorch implementation is
used when Triton/CUDA is unavailable.

**Measured result and tradeoff.** In a targeted large-batch ablation, partial
fusion reduced the compiled model from 32 to 29 GPU kernels and lowered latency
from `26.6799 ms` to `25.4092 ms` (`4.76%`). Across the same-host timeline, the
forward-order geometric mean moved from `11.676x` to `11.785x`, but the sub-1%
aggregate difference changed sign with measurement order. We retained the
partial fusion because it preserves exact semantics and the FP32 pre-GELU
boundary; we do not present it as a universal standalone speedup.

### 5. Exact memory-bounded scheduling for shape #14

**Bottleneck and insight.** Shape #14 has FP32 input and output tensors of
`[32,100000,1024]`, each occupying `12.207 GiB`. Flash attention removes the
approximately `18.6 TiB` score tensor, but a full-batch FP16 packed-QKV tensor
would still add `18.311 GiB`. Input, output, QKV, and workspace therefore cannot
coexist within 32 GiB. The key invariant is that batch samples never attend to
or otherwise interact with one another.

**Implementation.** Perry allocates the required full output once, processes
one sample through both Transformer layers, and immediately copies that result
into its output slice before moving to the next sample. This bounds temporary
QKV and attention storage to `B=1` while preserving the exact full-batch output
contract. The outer loop is marked `torch.compiler.disable` so Dynamo cannot
unroll 32 iterations into a memory-heavy graph. Its single-sample body is
compiled lazily once and reused; the cached executor is invalidated after
weight/device/mode changes.

**Correctness and measured result.** This is only a scheduling transformation:
no layer, weight, mask, arithmetic boundary, or output element is approximated.
Compared with the earlier eager sample schedule, compiling and reusing the
`B=1` body reduced optimized shape-#14 latency by `3.11–3.61%` and peak
allocation from `26.977` to `24.487 GiB`. The final full-output gate passed all
`3,276,800,000` elements. Its official latency is reported separately because
the original baseline remains physically inexecutable on the target GPU.

These excerpts are condensed from the active standalone implementation:

```python
# Flash-first mixed-precision causal attention.
q, k, v = (
    F.linear(x.to(dtype=torch.float16),
             attention._qkv_weight_mixed,
             attention._qkv_bias_mixed)
    .reshape(batch, seq_len, 3, attention.num_heads, attention.head_dim)
    .permute(2, 0, 3, 1, 4)
    .unbind(0)
)
if causal:
    with sdpa_kernel(backends=FLASH_FIRST_BACKENDS, set_priority=True):
        context = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=True,
            scale=attention.scale,
        )

# Memory-bounded outer schedule; the sample body is compiled and reused.
@torch.compiler.disable
def _forward_large_sequence(self, x, valid_token_mask):
    output = torch.empty_like(x)
    chunk_size = self._LARGE_SEQUENCE_BATCH_CHUNK
    for start in range(0, x.shape[0], chunk_size):
        end = min(start + chunk_size, x.shape[0])
        chunk_mask = None if valid_token_mask is None else valid_token_mask[start:end]
        output[start:end].copy_(
            self.forward_large_sequence_sample(x[start:end], chunk_mask)
        )
    return output
```

The full kernel and cache lifecycle are available in the
[`standalone final implementation`](https://github.com/wheres-my-perry/techjam-2026-track3/blob/main/v16_1_clean.py).

## Correctness and benchmark methodology

The organizer's implementation remained the oracle. Baseline and optimized
models received identical weights, inputs, masks, seed, public dtype, TF32
policy, and shape. Accuracy ran before performance; failed candidates were not
timed. Compilation and autotuning happened before the steady-state window.

The final environment was a Vast.ai Ubuntu 24.04.4 container with an AMD Ryzen
5 5600X CPU, 33,564,246,016 bytes of RAM, and an NVIDIA GeForce RTX 5090
(`sm120`, 32,607 MiB visible VRAM). It used driver 595.71.05, Python 3.12.14, PyTorch
2.11.0+cu128, CUDA 12.8, cuDNN 9.19.0, and Triton 3.6.0. Public tensors were
FP32, TF32 was enabled for both paths, matmul precision was `high`, and seed was
1234. The measured revision was `4f77a04fd51c3a2ad8b9d8986657915ae3ca94d6`.

### Official shapes #1–#13

The final run used five accuracy trials, followed by 20 warmups, 100 CUDA Event
repeats, and three rounds with alternating baseline/optimized order.

All rows below passed with zero failed elements. The worst maximum absolute
error across shapes #1–#13 was `0.00179085`; pass/fail still came from the full
absolute-OR-relative comparator, not this summary statistic.

All 13 shapes use four layers and `FFN=D`. The first column shows
`B×S×D / heads`, making each workload regime visible without requiring the
reader to cross-reference the problem statement.

| Shape and workload (`B×S×D / H`) | Baseline median | Optimized median | Speedup | Max absolute error |
|---:|---:|---:|---:|---:|
| #1 — `64×128×128 / 4` | 1.7640 ms | 0.1343 ms | 13.134x | 0.00127273 |
| #2 — `1×128×128 / 4` | 1.7764 ms | 0.1108 ms | 16.035x | 0.00114284 |
| #3 — `4×128×128 / 4` | 1.7882 ms | 0.1108 ms | 16.142x | 0.00124183 |
| #4 — `16×128×128 / 4` | 1.7384 ms | 0.1118 ms | 15.549x | 0.00134721 |
| #5 — `128×128×128 / 4` | 1.7535 ms | 0.2102 ms | 8.340x | 0.00147235 |
| #6 — `10000×128×128 / 4` | 177.3218 ms | 25.1813 ms | 7.042x | 0.00160612 |
| #7 — `64×128×32 / 4` | 1.7424 ms | 0.1118 ms | 15.584x | 0.00179085 |
| #8 — `64×128×1024 / 4` | 6.6464 ms | 2.7936 ms | 2.379x | 0.00134873 |
| #9 — `64×128×128 / 1` | 1.5747 ms | 0.1513 ms | 10.409x | 0.00126606 |
| #10 — `64×128×128 / 2` | 1.7550 ms | 0.1404 ms | 12.498x | 0.00134724 |
| #11 — `64×128×128 / 16` | 1.7345 ms | 0.1860 ms | 9.326x | 0.00140083 |
| #12 — `64×32×128 / 4` | 1.7501 ms | 0.1098 ms | 15.940x | 0.00134721 |
| #13 — `64×1024×128 / 4` | 41.8362 ms | 1.0793 ms | 38.762x | 0.00147235 |

**Geometric-mean speedup: 11.803x.** This is the predeclared final-submission
start control, not the best run selected after the fact. The repeated end
control measured `11.838x`; baseline/optimized geometric means and heavy shapes
#6/#8/#13 all stayed within the predeclared 3% drift budget.

The same source revision measured `7.904x` on an earlier driver-580 host. The
new host's baseline geomean was 72.48% slower while its optimized geomean was
15.51% slower, so the ratio change is a cross-host effect—not a code-improvement
claim.

The spread is itself an architectural result. Shape #13 has enough
sequence-length work to amortize compilation and benefits strongly from fused
attention, reaching `38.762x`. Shape #8 is dominated by wide `D=1024` GEMMs
that already use efficient library kernels, leaving less removable overhead and
therefore reaching `2.379x`. Shapes #2–#4 are launch-sensitive small batches,
while shape #6 is a 1.28-million-token projection/FFN workload. These regimes
explain why Perry uses workload-aware execution rather than one claimed
universally optimal kernel.

### Official shape #14

The original explicit-score reference is statically infeasible because its
attention tensor would require approximately 18.6 TiB. We therefore validated
correctness with a memory-bounded oracle over the complete `B=32` output. The
oracle preserves the reference formula, weights, FP32 softmax, masks, and
strict comparator, but streams query blocks and comparison-token blocks so the
score tensor never exists in full. A reduced-shape test first checked this
oracle against the original implementation before it was used for shape #14.
After accuracy passed, we timed the native full-output optimized forward with
one warmup and five CUDA Event repeats.

The result was **PASS, `0/3,276,800,000` failed elements**, with maximum absolute
error `0.000944197` and mean absolute error `0.0000656367`. The optimized forward
achieved a `6987.4644 ms` median, `6994.0999 ms` p90,
`457,962.98 token/s`, and `24.487 GiB` peak allocated memory. Baseline latency
and speedup are **N/A** because the original explicit-score reference cannot
execute within 32 GiB; we do not substitute the blocked correctness reference
as a performance baseline.

The environment manifest and final raw JSON, CSV, and logs are published in
[`results/final/`](https://github.com/wheres-my-perry/techjam-2026-track3/tree/main/results/final).
The complete same-host chronology is available in the
[`benchmark timeline report`](https://github.com/wheres-my-perry/techjam-2026-track3/blob/main/BENCHMARK_TIMELINE_REPORT.md),
with its
[`curated timeline artifacts`](https://github.com/wheres-my-perry/techjam-2026-track3/tree/main/results/timeline-rtx5090-driver595).

## Ideas explored but not included in the final solution

Not every idea outside the standalone final implementation was simply “bad.”
Some failed correctness, some were accurate but slower end to end, and others
remained promising prototypes without enough target-GPU evidence. The complete
experiment history is in the
[`technical report`](https://github.com/wheres-my-perry/techjam-2026-track3/blob/main/SOLUTION.md),
[`experiment log`](https://github.com/wheres-my-perry/techjam-2026-track3/blob/main/EXPERIMENTS.md),
and
[`attention research report`](https://github.com/wheres-my-perry/techjam-2026-track3/blob/main/ATTENTION_OPTIMIZATION_RESEARCH.md).

### Rejected by the correctness gate

- **More aggressive low precision.** BF16 internal compute, per-tensor FP8,
  Blackwell MXFP8, full FP16 accumulation, and symmetric INT8 FFN-in were all
  tested. Each exceeded the strict error budget in at least one gate; full FP16
  accumulation also showed no paired speed advantage. We therefore retained
  FP16 only inside the measured precision islands.
- **Quantized SageAttention.** A QK-INT8/PV-FP16 recipe was `1.393x` faster than
  PyTorch Flash in isolated long-sequence attention, but its full-Transformer
  `B=1` canary failed `1/102,400,000` output elements. An exact-prefix correction
  reduced local error yet still failed official shapes #6 and #9. A single
  failed element is enough to disqualify the route.

### Correct, but not a robust end-to-end improvement

- **Approximate GELU and residual/LayerNorm rewrites.** Tanh GELU passed all
  shapes #1–#13, but was `0.32%` slower in the clean paired shape-#8 run and
  disturbed an existing compiler fusion. The residual/LayerNorm rewrite was
  equivalent but produced no new code generation because TorchInductor already
  fused the relevant path.
- **Fully fused persistent MLP.** Keeping both FFN matrix multiplications and
  GELU in one persistent kernel produced `1.18–1.59x` isolated-kernel gains, but
  regressed whole-model latency on shapes #1, #5, and #13 and was slightly
  slower than the final partial fusion on shape #6. Fewer kernels did not mean
  a faster compiled Transformer.
- **Alternative attention backends.** On the exact shape-#14 inner workload,
  FlashAttention-4 passed correctness but was `7.72%` slower than PyTorch Flash;
  cuDNN was `2.38–2.92%` slower and the memory-efficient backend was nearly
  twice as slow. A hard-coded per-shape backend table was also replaced by the
  simpler Flash-first priority.
- **Direct-layout QKV.** Writing contiguous Q/K/V directly improved shape #13
  by `0.98–2.20%` and shape #6 by about `3.43%`, but 11 of the other 12 probes
  showed no robust win. The submission avoids an exact-test-tuple branch until
  a general workload-based router is validated.
- **Checkpointed FP16 WMMA.** This later FFN-in prototype passed the complete
  correctness gate, but its #1–#13 geometric-mean speedup was `10.3079x` versus
  the submitted path's `11.8030x`; direct optimized latency regressed `13.73%`.
  Correctness alone was not enough to promote it.
- **Parallel long-sequence scheduling.** A four-partition follow-up using the
  submitted arithmetic passed all `3.2768B` shape-#14 elements and reached
  `6780.3867 ms`, about `1.51–1.66%` faster than its single-partition controls.
  It remains outside `main.py`: the gain is small and shape-specific, requires
  disabling shared CUDA Graph replay, uses `25.676 GiB` peak allocation in that
  mode, and still needs broader repeatability and portability evidence.

### Promising, but not sufficiently validated

- **Precision and kernel prototypes.** Direct FP32 FFN-out storage improved a
  local accuracy canary but lacks a full timing matrix and increases memory
  traffic. A persistent FFN-in scheduler also has no target-GPU result.
- **Exact-attention and fusion research.** FlashInfer SM120, Transformer Engine
  LayerNorm-linear/MLP fusion, cuBLASLt/CUTLASS epilogues, no-concat QKV, causal
  load balancing, and a dedicated `head_dim=8` kernel remain unshipped because
  none has completed the exact-workload correctness and end-to-end timing gate.
  Sparse, linear, low-rank, decode/KV-cache, and distributed attention target
  different semantics or workloads and are not substitutes for this task.

These results reinforced one rule: a faster microkernel, a lower precision, or
fewer launches is only useful when the complete model still passes correctness
and wins under the same end-to-end protocol.

## Development tools, APIs, libraries, datasets, and assets

- **Development tools:** OpenAI Codex, Git/GitHub, SSH, Python CLI, PyTorch
  Profiler/Kineto, and CUDA Events.
- **APIs:** PyTorch eager, SDPA, `torch.compile`, `torch.library.custom_op`, and
  Triton kernel APIs. No external service API is called at runtime.
- **Libraries and frameworks:** PyTorch, Triton, and CUDA/cuDNN through PyTorch.
- **Datasets:** none; the benchmark generates synthetic tensors from a fixed
  seed.
- **Assets:** the organizer-provided PyTorch benchmark, the 14 published test
  shapes, and public technical documentation.

## Impact and relevance

Perry demonstrates a reusable GPU-engineering workflow: establish a trusted
oracle, profile the workload, specialize with evidence, retain fallbacks, and
publish negative results. The project optimizes computation, layout, precision,
compiler scope, and tensor lifetime as one system rather than treating kernel
speed as the only metric. Shape #14 is the clearest example: the mathematical
operation was valid, but only a different execution schedule made it physically
executable. The standalone artifact still preserves PyTorch parameter names,
strict `state_dict` loading, and the original forward contract.

## Limitations

Measurements target one RTX 5090/SM120 stack; other GPUs require retuning.
Mask elision assumes right padding. The fast path targets FP32 inference with
internal FP16 compute, while training and unsupported dtypes fall back. Shape
#14 has no paired baseline latency, the final gate used seed 1234, and reported
steady-state latency excludes compile/autotune cold start.

## Future work

Our roadmap comes from the measured follow-up and candidate catalogue in the
[`attention optimization research report`](https://github.com/wheres-my-perry/techjam-2026-track3/blob/main/ATTENTION_OPTIMIZATION_RESEARCH.md).
It is deliberately ordered by evidence rather than novelty. Every candidate
would keep the same strict comparator and public interface, and would need an
end-to-end win after layout, quantization, compilation, and memory costs—not
just a faster isolated kernel.

1. **Attack exact long-sequence attention first.** Attention consumes `92.258%`
   of shape #14's compiled-executor device time. The next library candidate is
   FlashInfer's SM120 prefill FMHA, measured first on shape #13 and then on the
   `B=1` shape-#14 body with every adapter and layout conversion included. If
   available libraries reach a ceiling, the next step is a custom exact SM120
   online-softmax kernel with causal triangular load balancing—not an
   approximate attention replacement.
2. **Fuse the path into attention and reduce its live set.** Prototype
   `LayerNorm → QKV projection → backend-native layout` using Transformer
   Engine or a CUTLASS epilogue. For shape #14, avoid a full packed activation by
   writing separate backend-ready Q/K/V buffers, reusing scratch storage, and
   choosing the batch chunk from measured memory headroom.
3. **Build an accuracy-aware workload router.** Autotune launch-bound, GEMM-heavy,
   standard-attention, long-attention, and extreme-memory regimes. Direct-layout
   QKV for large `B*S` with `D=FFN=128` is the first evidence-backed route because
   it already helped shapes #6 and #13. Routing keys must describe workload and
   environment properties, never an official test ID.
4. **Raise the validation and portability bar.** Repeat correctness across more
   seeds, input scales, padding ratios, causal/non-causal inputs, and head
   dimensions, then repeat timing on another GPU/software stack. Record the
   backend that actually ran, allocated and reserved memory, compile cold start,
   and steady-state latency. The measured four-partition shape-#14 scheduler is
   a promotion candidate only after these repeatability checks justify its extra
   stream and CUDA-Graph complexity.
5. **Specialize remaining bottlenecks only after profiling.** For shapes #6 and
   #8, compare exact cuBLASLt GELU epilogues, Transformer Engine MLP fusion, and
   CUTLASS SM120 schedules with the current Triton path. For `head_dim=8` or
   `D=32`, evaluate a small persistent attention/block kernel. Protected
   INT8/FP8 precision islands with smoothing and FP32 outlier correction remain
   later accuracy-first experiments because earlier quantized paths failed.

We would not spend additional time on semantics-changing sparse/linear
attention, decode-oriented kernels, or larger batch chunks without new
evidence. They either violate the required dense self-attention computation or
have already failed to move the measured bottleneck enough to justify their
complexity.

## Reproduction

Public repository:
[wheres-my-perry/techjam-2026-track3](https://github.com/wheres-my-perry/techjam-2026-track3).

```bash
# Official shapes #1–#13 using the historical-checkpoint adapter.
CUDA_VISIBLE_DEVICES=0 python3 timeline_runner.py --checkpoints v16_1 \
  --shape-ids 1-13 \
  --device cuda:0 --dtype float32 --accuracy-trials 5 \
  --warmup 20 --repeats 100 --benchmark-rounds 3 \
  --compile-mode max-autotune

# Shape #14: reference feasibility record and final-submission gates/timing.
CUDA_VISIBLE_DEVICES=0 python3 shape14_timeline_runner.py \
  --checkpoints baseline,v16_1 --device cuda:0 --batch-limit 32 \
  --query-chunk 256 --compare-token-chunk 2048 --seed 1234 \
  --warmup 1 --repeats 5 --compile-mode max-autotune
```

The complete technical report and experiment history are available in
[`SOLUTION.md`](https://github.com/wheres-my-perry/techjam-2026-track3/blob/main/SOLUTION.md)
and
[`EXPERIMENTS.md`](https://github.com/wheres-my-perry/techjam-2026-track3/blob/main/EXPERIMENTS.md).

## Team contributions

- **Le Tuan Hoang:** Coordinated GPU access, contributed the high-level attention
  algorithm direction, FP32 pre-GELU accumulation,
  and proposed SDPA and FP32-to-FP16 precision reduction.
- **Vo Khac Trieu:** Owned the end-to-end technical implementation. He turned
  the SDPA and mixed-precision suggestions into working code, then designed and
  implemented the remaining optimizations: Flash-first attention, Triton
  FFN/GELU fusion, and memory-bounded long-sequence
  scheduling. He also built the benchmark tooling, drove the AI-assisted
  development loop, and ran and analyzed all correctness and performance tests.
- **Le Kien Thanh and Nguyen An Thinh:** Produced the slides and demo video.
  Their Track 3 work focused on presentation while they primarily handled the
  team's Track 5 project.

## Demo

**TODO(owner): Add the public YouTube URL.**
