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
The winning strategy was to optimize at three levels while preserving one
public interface, `forward(x, valid_token_mask)`:

1. Reduce graph, data-movement, and launch overhead using packed projections,
   Flash-first SDPA, custom fusion, and whole-model compilation.
2. Use lower precision only where Tensor Core throughput helps, while keeping
   numerically sensitive boundaries in FP32.
3. Change the execution schedule when tensor lifetime—not the Transformer
   formula—is the reason a valid workload cannot fit in memory.

The result was an accuracy-gated loop: profile, test one hypothesis, reject
failures, and promote only measured end-to-end wins.

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

The fast path targets FP32 inference with internal FP16 compute. Non-causal
cases retain their key mask and use automatic SDPA; training and unsupported
public dtypes use reference arithmetic. The backend priority also falls through
to supported implementations when Flash is unavailable. Parameter names remain
compatible with the reference `state_dict`; derived FP16 packed weights are
non-persistent caches that refresh after `load_state_dict()` and device/dtype
moves.

## Algorithm and implementation details

### 1. Flash-first attention with a mask proof

The reference materializes `QKᵀ`, mask operations, softmax probabilities, and
the probability–V product. Perry instead calls PyTorch SDPA with the backend
priority Flash → cuDNN → memory-efficient → math.

For causal self-attention with right padding, valid tokens form a prefix. A
valid query can only attend to keys at its own or earlier position, so it cannot
observe padded keys in the suffix. The causal fast path can therefore omit the
redundant key mask and zero invalid query outputs at the block boundary.
Non-causal execution keeps the key mask. This proof does not apply to arbitrary
sparse masks, which remain outside the optimized mask-elision contract.

### 2. Mixed precision with FP32 safety boundaries

Public inputs and outputs, LayerNorm, residual accumulation, and the custom
FFN-in dot-product accumulator remain FP32. QKV, SDPA, projections, and FFN
GEMMs use FP16 operands for Tensor Core throughput. The exact GELU reads
`accumulator + bias` in FP32 before its output is stored in FP16 for FFN-out.
This removes an unnecessary pre-GELU rounding boundary without adding another
intermediate tensor or launch.

### 3. Custom Triton FFN/GELU fusion

The first FFN matrix multiplication, bias, and exact erf-GELU are fused in a
Triton operator. `torch.library.custom_op` and a fake implementation allow
TorchInductor to capture the kernel inside the compiled graph. In the same-host
timeline, adding this fusion moved the forward-order geometric mean from
`11.676x` to `11.785x`, while the reverse order changed the comparison's sign.
We therefore treat it as an architectural fusion with shape-dependent value,
not as a robust aggregate speedup claim.

### 4. Exact memory-bounded scheduling for shape #14

The shape-#14 input and output each occupy 12.207 GiB in FP32, while full-batch
packed QKV would add 18.311 GiB. Batch samples do not interact, so Perry
preallocates the final output, processes one sample through all layers, and
writes it into the corresponding output slice. The outer loop remains eager to
prevent Dynamo from unrolling 32 samples into one graph; the `B=1` body is
compiled once and reused. This changes live tensor lifetime and launch
scheduling, not the Transformer computation.

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

| Shape | Baseline median | Optimized median | Speedup | Max absolute error |
|---:|---:|---:|---:|---:|
| 1 | 1.7640 ms | 0.1343 ms | 13.134x | 0.00127273 |
| 2 | 1.7764 ms | 0.1108 ms | 16.035x | 0.00114284 |
| 3 | 1.7882 ms | 0.1108 ms | 16.142x | 0.00124183 |
| 4 | 1.7384 ms | 0.1118 ms | 15.549x | 0.00134721 |
| 5 | 1.7535 ms | 0.2102 ms | 8.340x | 0.00147235 |
| 6 | 177.3218 ms | 25.1813 ms | 7.042x | 0.00160612 |
| 7 | 1.7424 ms | 0.1118 ms | 15.584x | 0.00179085 |
| 8 | 6.6464 ms | 2.7936 ms | 2.379x | 0.00134873 |
| 9 | 1.5747 ms | 0.1513 ms | 10.409x | 0.00126606 |
| 10 | 1.7550 ms | 0.1404 ms | 12.498x | 0.00134724 |
| 11 | 1.7345 ms | 0.1860 ms | 9.326x | 0.00140083 |
| 12 | 1.7501 ms | 0.1098 ms | 15.940x | 0.00134721 |
| 13 | 41.8362 ms | 1.0793 ms | 38.762x | 0.00147235 |

**Geometric-mean speedup: 11.803x.** This is the predeclared final-submission
start control, not the best run selected after the fact. The repeated end
control measured `11.838x`; baseline/optimized geometric means and heavy shapes
#6/#8/#13 all stayed within the predeclared 3% drift budget.

The same source revision measured `7.904x` on an earlier driver-580 host. The
new host's baseline geomean was 72.48% slower while its optimized geomean was
15.51% slower, so the ratio change is a cross-host effect—not a code-improvement
claim.

### Official shape #14

The original explicit-score reference is statically infeasible because its
attention tensor would require approximately 18.6 TiB. We therefore validated
correctness with a memory-bounded oracle over the complete `B=32` output, then
timed the native optimized forward with one warmup and five CUDA Event repeats.

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
  tested. Each exceeded the unchanged strict error budget in at least one
  official gate. Full FP16 accumulation also provided no paired speed advantage
  on the valid control, while even weight-only INT8 failed the smallest scoped
  probe. We therefore retained FP16 operands only inside the measured precision
  islands, with FP32 at sensitive boundaries.
- **Quantized SageAttention.** A QK-INT8/PV-FP16 recipe was `1.393x` faster than
  PyTorch Flash in isolated long-sequence attention, but its full-Transformer
  `B=1` canary failed `1/102,400,000` output elements. An exact-prefix correction
  reduced the local error yet still failed official shapes #6 and #9. Its
  automatic INT8/FP8 route remains a performance-only diagnostic, not a valid
  submission candidate.

### Correct, but not a robust end-to-end improvement

- **Approximate GELU and residual/LayerNorm rewrites.** Tanh GELU passed all
  shapes #1–#13, but was `0.32%` slower in the clean paired shape-#8 run and
  disturbed an existing compiler fusion. The residual/LayerNorm pipeline was
  equivalent, but TorchInductor already generated the intended fused graph; it
  was also `0.32%` slower on shape #8. A custom replacement would have added
  complexity without removing a measured bottleneck.
- **Fully fused persistent MLP.** Keeping both FFN matrix multiplications and
  GELU in one persistent kernel produced `1.18–1.59x` isolated-kernel gains, but
  regressed whole-model latency on shapes #1, #5, and #13 and was slightly
  slower than the final partial fusion on shape #6. This is why kernel-count
  reduction alone was not a promotion criterion.
- **Alternative attention backends.** On the exact shape-#14 inner workload,
  FlashAttention-4 passed correctness but was `7.72%` slower than PyTorch Flash;
  cuDNN was `2.38–2.92%` slower and the memory-efficient backend was nearly
  twice as slow. A hard-coded per-shape SDPA table was also superseded by the
  simpler Flash-first priority because the static mapping did not improve the
  aggregate result enough to justify test-shaped source logic.
- **Direct-layout QKV and larger long-sequence chunks.** Writing Q/K/V directly
  in attention-native layout improved shape #13 by `0.98–2.20%` and shape #6 by
  about `3.43%`, but 11 of the other 12 cross-shape probes did not show a robust
  win. The final source therefore avoids an exact-test-tuple branch until a
  general workload-based router is validated. Increasing the long-sequence
  executor from `B=1` to `B=2` also passed full shape-#14 correctness, but gained
  only `0.30–0.59%` with unchanged peak allocation—too close to measurement
  drift to outweigh the simpler `B=1` schedule.

### Promising, but not sufficiently validated

- **Precision-boundary and scheduling prototypes.** Direct FP32 FFN-out storage
  improved local numerical margin and passed a CUDA shape-#7 accuracy canary,
  but its full performance matrix was not completed and the wider output
  increases memory traffic. A persistent FFN-in scheduler had no measured
  result. Checkpointed-FP16 WMMA accumulation and multi-stream batch partitions
  passed local structural tests only; CUDA compilation, full correctness,
  memory, and paired timing gates remain open.
- **Research-only directions.** FlashInfer SM120, Transformer Engine
  LayerNorm-linear/MLP fusion, cuBLASLt/CUTLASS epilogues, a dedicated
  `head_dim=8` kernel, no-concat QKV for shape #14, two-dimensional
  batch/sequence streaming, causal load balancing, accuracy-aware autotuning,
  and outlier-correction paths were not shipped because they lack complete
  exact-workload validation. FlashAttention-3 targets Hopper rather than the
  RTX 5090's `sm120`, while Transformer Engine's FP8 attention path was not
  eligible on `sm120`. Sparse, linear, and low-rank attention would change the
  required dense semantics; MLA, KV-cache/decode, distributed, and CPU kernels
  target different workloads.

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
publish negative results. Shape #14 shows that rescheduling tensor lifetimes can
make valid mathematics executable. The standalone artifact preserves PyTorch
parameter names, strict `state_dict` loading, and the original forward contract.

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

1. **Attack exact long-sequence attention first.** Attention already consumes
   `92.258%` of shape #14's compiled-executor device time. The first unmeasured
   candidate should be FlashInfer's SM120 prefill FMHA, tested on shape #13 and
   then the `B=1` shape-#14 body with every adapter and layout conversion inside
   the timed region. PyTorch Flash remains the control: cuDNN, the
   memory-efficient backend, and FlashAttention-4 have already lost this exact
   comparison. Only if available libraries hit a ceiling would we build a
   custom exact SM120 kernel using online softmax and causal triangular
   load-balancing for `S=100000`.
2. **Fuse the path into attention and reduce live memory.** Prototype
   `LayerNorm → QKV projection → backend-native layout` with Transformer Engine
   `LayerNormLinear` or a CUTLASS epilogue. For shape #14, retain packed weights
   but avoid a full packed activation: write separate contiguous Q/K/V buffers
   that FlashInfer or another exact backend can consume directly, reuse a
   scratch arena, and select the batch chunk from measured headroom. This targets
   both layout traffic and the 18.311 GiB full-batch packed-QKV pressure.
3. **Build an accuracy-aware workload router.** Autotune distinct regimes for
   launch-bound shapes, GEMM-heavy shapes, standard attention, long attention,
   and extreme-memory execution. A direct-layout QKV route for large `B*S` with
   `D=FFN=128` is the first candidate because it already showed measured wins on
   shapes #6 and #13. Routing must use meaningful workload and environment keys,
   never an official test ID, and a route becomes eligible only after its full
   accuracy matrix passes without reducing the aggregate score.
4. **Strengthen the measurement and portability gate.** Expand correctness to
   multiple seeds, input scales, padding ratios, causal/non-causal execution,
   every supported head dimension, and a second GPU/software stack. Record the
   SDPA backend that actually ran, peak allocated and reserved memory, and both
   compile cold-start and steady-state latency. Compiled artifacts and autotune
   choices should be cached by GPU, driver, CUDA, PyTorch, dtype, shape, and mask
   contract rather than assumed portable.
5. **Explore lower precision only as protected precision islands.** The next
   quantized-attention experiment would use SageAttention2++-style QK INT8 with
   smoothing, PV in FP16 with two-level/FP32 accumulation, and FP32 softmax,
   projection, and residual boundaries. Outlier rows or channels would receive
   a small FP32 correction. Because earlier Sage variants missed the strict
   gate, this path starts with accuracy-only canaries on shapes #1, #8, and #13;
   quantization and smoothing overhead enter timing only after correctness.
6. **Specialize the remaining proven bottlenecks.** For GEMM-heavy shapes #6
   and #8, compare exact cuBLASLt GELU epilogues, Transformer Engine
   `LayerNormMLP`, and CUTLASS SM120 schedules against the current Triton path.
   For `head_dim=8` and `D=32` shapes, test a small persistent attention/block
   kernel that avoids wasteful general-purpose tiling. These are later-stage
   projects: each begins with a fresh profile, isolates one change, and must beat
   the whole model—not merely a microbenchmark.

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
