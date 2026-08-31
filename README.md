# Perry — Accuracy-Gated Transformer Optimization on RTX 5090

## Project overview

Perry is a standalone, AI-assisted PyTorch Transformer implementation for
TikTok TechJam 2026 Track 3. It passes the strict correctness gate on all 14
official shapes, reduces inference latency on an NVIDIA GeForce RTX 5090, and
makes the extreme `B=32, S=100000, D=1024` shape #14 executable within 32 GiB
of GPU memory. The implementation preserves the public interface
`UserOptimizedTransformer.forward(x, valid_token_mask)`, the output shape
`[batch_size, seq_len, d_model]`, and the contest correctness rule:

```text
relative_error < 0.02 OR absolute_error < 0.002
```

Both comparisons are strict and correctness is checked before performance.

The frozen submission is identified by tag `techjam-2026-final-v16.1` and the
machine-checked source hashes in [`SUBMISSION.md`](SUBMISSION.md). Start there
for the shortest clean-clone verification path.

### Who this helps and why it matters

Perry is intended for GPU inference and ML-systems engineers evaluating dense
Transformer encoder-style layers on memory-constrained NVIDIA GPUs. The direct
value is both lower latency and greater workload capacity: official shapes
#1-#13 achieved an `11.803x` same-host geometric-mean speedup with zero strict
correctness failures, while shape #14 became executable within a 32 GiB GPU at
`24.487 GiB` peak memory and `457,962.98` tokens/s.

Adoption is intentionally low-friction for a benchmark-compatible layer. The
optimized module keeps the same `forward(x, valid_token_mask)` API, output shape,
parameter names, and strict `state_dict` loading, and it retains safe fallbacks
outside the specialized inference path. See [Standalone use](#standalone-use)
for the integration example.

The evidence boundary is important: these are validated layer-level results on
the 14 official configurations, not a measured end-to-end production model,
cost, or energy result. Recommendation, retrieval/reranking, vision, speech, and
NLP encoders are intended evaluation targets for future integration work rather
than claimed deployments.

### Active implementation

The final submission uses one active implementation:

- `v16_1_clean.py` — standalone V16.1 model, config, FP16 inference cache,
  Flash-first attention, Triton FP32-pre-GELU kernel and memory-bounded compiled
  executor for long sequences.
- `main.py` — thin adapter that connects the standalone class to the official
  benchmark CLI.

`v16_1_clean.py` imports no benchmark code and no earlier implementation file.
Its only runtime dependencies are PyTorch and optional Triton. Historical
`v1`–`v18` files are preserved under `archive/versions/`; they are not active
runner aliases. Their results and decisions remain in `EXPERIMENTS.md`,
`SOLUTION.md` and `DECISION.md`. Unpromoted V19 research prototypes live under
`candidates/v19/`; they are not imported by `main.py` or used for the final
results.

The active algorithm keeps LayerNorm, residuals and public output in FP32;
uses FP16 operands for QKV, SDPA, projections and FFN GEMMs; evaluates exact
GELU from the FP32 FFN-in accumulator; and uses a reusable compiled B=1 body
inside an eager batch loop when `B > 1` and `S >= 8192`.

## Final validated results

The active standalone artifact was revalidated on commit
`4f77a04fd51c3a2ad8b9d8986657915ae3ca94d6` in a Vast.ai Ubuntu 24.04.4
container with an AMD Ryzen 5 5600X, 33.56 GB RAM, an RTX 5090 and NVIDIA
driver `595.71.05`. The run used PyTorch `2.11.0+cu128`, CUDA `12.8`, cuDNN
`9.19.0`, Triton `3.6.0`, public FP32 tensors, TF32 enabled and
`torch.compile(mode="max-autotune")` for shapes #1–#13.

- Official #1–#13: all shapes strict PASS, zero failed elements, worst max
  absolute error `0.00179085`, and predeclared start-control geometric-mean
  speedup **11.803x**. End-control measured `11.838x`; the 3% drift gate PASS.
- Official #14: full `B=32` strict PASS, `0/3,276,800,000` failed elements,
  max absolute error `0.000944197`.
- Shape #14 native B32 PASS; optimized-only median: `6987.4644 ms`, throughput
  `457,962.98 token/s`, peak allocated memory `24.487 GiB`.
- Shape #14 baseline latency and speedup are N/A because the original reference
  would materialize an approximately `18.6 TiB` attention-score tensor.

The commands, complete per-shape table, environment manifest and raw evidence
are checked in under [`results/final/`](results/final/README.md).
The full 11-checkpoint timeline and reverse-order repeats are under
[`results/timeline/`](results/timeline/README.md).
The complete execution report, including all checkpoint/shape speedups,
correctness failures, drift controls, source hashes and shape-#14 stages, is in
[`docs/benchmark-timeline/REPORT.md`](docs/benchmark-timeline/REPORT.md).
Earlier driver-580 evidence remains archived under
[`results/archive/cross-host-driver580/`](results/archive/cross-host-driver580/README.md). The
ratio change from `7.904x` to `11.803x` is not a code improvement: the new
host's baseline geomean was `72.48%` slower while optimized geomean was
`15.51%` slower.

## Setup and installation

Prerequisites:

- Linux with an NVIDIA GPU and a driver compatible with CUDA 12.8.
- Python 3.12 and `venv`.
- Enough GPU memory for the selected shape. Full shape #14 was validated on a
  32 GiB RTX 5090.

Clone the public repository and create an isolated environment:

```bash
git clone https://github.com/wheres-my-perry/techjam-2026-track3.git
cd techjam-2026-track3
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The pinned environment reproduces the final stack: PyTorch `2.11.0+cu128`,
CUDA `12.8` and Triton `3.6.0`. Confirm that PyTorch sees the GPU:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## Steps to reproduce the results

The commands below mirror the final single-GPU Vast.ai environment and use
`CUDA_VISIBLE_DEVICES=0`. On the two-GPU development host, the project is
assigned physical GPU index `1`; use `CUDA_VISIBLE_DEVICES=1` there. In both
cases PyTorch addresses the one visible GPU as `cuda:0`.

### 1. Run the preflight checks

```bash
python3 tools/submission_preflight.py
python3 -m py_compile main.py tools/matrix_runner.py tools/profile_models.py \
  torch_transformer_benchmark.py v16_1_clean.py tools/shape14/accuracy.py \
  tools/shape14/optimized_benchmark.py tools/shape14/profile.py \
  tools/shape14/fa4_probe.py tools/shape14/sage_probe.py tools/timeline_adapter.py \
  tools/timeline_runner.py tools/shape14/checkpoint_worker.py tools/shape14/timeline_runner.py
python3 -m tools.matrix_runner --list-shapes
python3 -m tools.profile_models --list-shapes
python3 -m tools.timeline_runner --list-checkpoints
python3 -m tools.shape14.timeline_runner --list-checkpoints
```

On the target GPU, verify the exact final stack and run a strict CUDA
correctness smoke before a long benchmark:

```bash
CUDA_VISIBLE_DEVICES=0 python3 tools/gpu_preflight.py \
  --device cuda:0 --strict-final-environment --require-idle
```

### 2. Run a short official-shape smoke test

```bash
CUDA_VISIBLE_DEVICES=0 python3 main.py \
  --device cuda:0 --dtype float32 \
  --batch-size 64 --seq-len 128 --d-model 128 \
  --heads 4 --ffn-dim 128 --layers 4 --causal
```

### 3. Reproduce official shapes #1–#13

```bash
CUDA_VISIBLE_DEVICES=0 python3 -m tools.timeline_runner \
  --checkpoints v16_1 \
  --shape-ids 1-13 \
  --device cuda:0 --dtype float32 \
  --accuracy-trials 5 \
  --warmup 20 --repeats 100 --benchmark-rounds 3 \
  --compile-mode max-autotune --timeout 1800
```

The runner checks accuracy before timing, isolates each shape in a subprocess,
and writes JSON/CSV results incrementally. On the final environment, V16.1
passed all 13 shapes with zero failed elements. The promoted `11.803x` headline
comes from the predeclared V16.1 start control in the full chronological sweep,
not from selecting the fastest repeat after the fact.

### 4. Reproduce official shape #14

Shape #14 uses separate tools because the supplied reference would materialize
an approximately 18.6 TiB attention-score tensor. Run the full memory-bounded
correctness check first:

```bash
CUDA_VISIBLE_DEVICES=0 python3 -m tools.shape14.timeline_runner \
  --checkpoints baseline,v16_1 --device cuda:0 --seed 1234 \
  --batch-limit 32 --query-chunk 256 --compare-token-chunk 2048 \
  --warmup 1 --repeats 5 --compile-mode max-autotune
```

The expected final result is strict PASS with `0/3,276,800,000` failed output
elements. The runner reports optimized-only latency after its correctness
stages; baseline latency and speedup remain N/A. Exact commands from the final
single-GPU machine, raw
logs, JSON/CSV outputs and the environment manifest are in
[`results/final/`](results/final/README.md).

### 5. Reproduce the full optimization timeline

The same-host cumulative sweep, including start/end drift controls, uses:

```bash
CUDA_VISIBLE_DEVICES=0 python3 -m tools.timeline_runner \
  --checkpoints v16_1,baseline,v1,v2,v3_1_eager,v3_1_compiled,v4_1,v4_2,v4_3,v8,v11,v16_1 \
  --shape-ids 1-13 \
  --device cuda:0 --dtype float32 --accuracy-trials 5 \
  --warmup 20 --repeats 100 --benchmark-rounds 3 \
  --compile-mode max-autotune --timeout 1800 \
  --control-drift-threshold 0.03
```

This is the run that produced the cumulative checkpoint table, the `11.803x`
start control, the `11.838x` end control and the passing 3% drift gate. See
[`docs/benchmark-timeline/REPORT.md`](docs/benchmark-timeline/REPORT.md) for the full
forward/reverse results and the failed-candidate audit trail.

## Standalone use

```python
import torch
from v16_1_clean import TransformerConfig, UserOptimizedTransformer

config = TransformerConfig(
    batch_size=64,
    seq_len=128,
    d_model=128,
    num_heads=4,
    ffn_dim=128,
    num_layers=4,
    causal=True,
)

model = UserOptimizedTransformer(config)
model.load_state_dict(official_state_dict, strict=True)
model = model.to(device="cuda:0", dtype=torch.float32).eval()
output = model(x, valid_token_mask)
```

For shapes below the long-sequence cutoff, the caller may optionally wrap the
whole model with `torch.compile(model, mode="max-autotune")`. V16.1 manages its
own compiled inner executor for the long-sequence path.

The causal fast path assumes right padding: each mask row is a `True` prefix
followed by a `False` suffix. Arbitrary sparse masks do not satisfy the proof
used to omit the causal key mask.

## Additional validation and profiling

Run the active syntax and orchestration checks:

```bash
python3 -m py_compile main.py tools/matrix_runner.py tools/profile_models.py \
  torch_transformer_benchmark.py v16_1_clean.py tools/shape14/accuracy.py \
  tools/shape14/optimized_benchmark.py tools/shape14/profile.py \
  tools/shape14/fa4_probe.py tools/shape14/sage_probe.py
python3 -m tools.matrix_runner --list-shapes
python3 -m tools.profile_models --list-shapes
```

The standalone cleanup has passed local strict state-dict checks, bitwise
causal/non-causal × mask/no-mask equivalence with the composed V16.1, training
and BF16 fallback equivalence, plus compiled-executor reuse/invalidation checks.
The fresh CUDA matrix on the standalone file also passed all official shapes:
#1–#13 used five accuracy trials and paired `20/100/3` timing, while #14 used
the full 32-batch streaming correctness gate and an optimized-only five-repeat
CUDA Event benchmark. See [`results/final/`](results/final/README.md).

## AI-assisted development and technical stack

- **AI use:** OpenAI Codex helped audit the statement and comparator, classify
  the official shapes, propose and implement PyTorch/Triton experiments,
  interpret profiler evidence, and maintain the benchmark/documentation trail.
  Every promoted change was still validated by the unchanged correctness gate
  and target-GPU measurements.
- **Runtime APIs:** PyTorch eager, scaled dot-product attention,
  `torch.compile`, `torch.library.custom_op`, and Triton kernel APIs. The
  submitted runtime calls no OpenAI API or other external service.
- **Tools:** Git/GitHub, SSH, Python CLI, PyTorch Profiler/Kineto and CUDA
  Events.
- **Data and assets:** no external dataset is used. Inputs are synthetic tensors
  generated from a fixed seed; the organizer-provided benchmark and 14 official
  shapes are the only project assets.

## Repository map

| Path | Role |
|---|---|
| `SUBMISSION.md` | Frozen release identity and clean-clone verification path |
| `submission-manifest.json` | Machine-readable hashes and evidence expectations |
| `v16_1_clean.py` | Only active optimized implementation |
| `main.py` | Official single-shape benchmark adapter |
| `torch_transformer_benchmark.py` | Baseline/reference oracle |
| `tools/matrix_runner.py` | Isolated runner for the 14 official shapes |
| `tools/profile_models.py` | Accuracy, timing and profiler runner |
| `tools/shape14/accuracy.py` | Memory-bounded strict accuracy for shape #14 |
| `tools/shape14/optimized_benchmark.py` | Optimized-only shape-#14 timing |
| `tools/timeline_adapter.py` | Archived-checkpoint registry, injection and preflight |
| `tools/timeline_runner.py` | Full #1–#13 checkpoint sweep and drift controls |
| `tools/shape14/timeline_runner.py` | Isolated Baseline/V16.1 shape-#14 stages |
| `tools/submission_preflight.py` | Standard-library source/evidence lock verifier |
| `tools/gpu_preflight.py` | CUDA environment, idle-state and correctness preflight |
| `docs/benchmark-timeline/REPORT.md` | Complete driver-595 benchmark execution report |
| `docs/research/attention-optimization.md` | Evidence-ranked attention follow-up catalogue |
| `docs/DEVPOST.md` | Submission-ready project narrative and team contribution record |
| `results/final/` | Checked-in final environment and benchmark evidence |
| `results/timeline/` | Curated timeline and reverse-order evidence |
| `results/experiments/` | Curated evidence for unpromoted candidates |
| `candidates/v19/` | Unpromoted scheduling/precision research prototypes |
| `archive/versions/` | Historical implementation and opcheck files |
| `runs/` | Gitignored generated benchmark, profile, trace and temporary output |
| `tests/` | Import, state-dict and mask-path smoke tests |
| `STATEMENT.md` | Competition statement and official shapes |
| `ARCHITECTURE.md` | Runtime and repository architecture |
| `EXPERIMENTS.md` | Commands and measured experiment log |
| `SOLUTION.md` | Full technical report |
| `DECISION.md` | Long-term technical decisions |
| `IMPLEMENTATION_PLAN.md` | Current phase and remaining work |
| `LICENSE` / `NOTICE.md` | Project license and third-party scope |

## Reflection: limitations and future improvements

- Final measurements target one RTX 5090/SM120 software stack; backend and
  compile choices must be retuned for other GPUs.
- The causal fast path assumes right padding rather than an arbitrary sparse
  token mask.
- Shape #14 has no paired baseline latency because the supplied reference would
  require an approximately 18.6 TiB score tensor.
- The official final gate uses seed 1234. More seeds, input scales, padding
  ratios and compile cold-start measurements would strengthen robustness.
- The optimized fast path targets FP32 inference with internal FP16 compute;
  training and unsupported public dtypes use the safer reference fallback.
- Reported latency is steady-state and excludes compilation and autotuning
  cold-start cost.

The solution deliberately favors measured, accuracy-valid end-to-end gains over
isolated kernel wins. This makes the final artifact reliable on the official
matrix, but it is not yet a universal Transformer runtime. With more time, the
next priorities would be exact FlashInfer SM120 attention, fusion of LayerNorm
into QKV projection and backend-native layout, a workload-based direct-QKV
router for large token volumes, then broader multi-seed, multi-hardware and
cold-start/backend validation. Accuracy-protected
low-precision islands and custom SM120 attention remain deferred until the
exact library and layout paths have been measured end to end. The
evidence-ranked roadmap is in
[`docs/research/attention-optimization.md`](docs/research/attention-optimization.md), and
the detailed reflection is in [`SOLUTION.md`](SOLUTION.md).

## Team member contributions

- **Le Tuan Hoang** — coordinated GPU access; contributed the high-level
  attention direction, FP32 pre-GELU accumulation, and the SDPA and FP32-to-FP16
  precision-reduction proposals.
- **Vo Khac Trieu** — owned the end-to-end technical implementation; implemented
  the SDPA and mixed-precision proposals, Flash-first attention, Triton
  FFN/GELU fusion and memory-bounded long-sequence scheduling; built the
  benchmark tooling and ran and analyzed the correctness/performance tests.
- **Le Kien Thanh and Nguyen An Thinh** — produced the slides and demo video;
  their Track 3 work focused on presentation while they primarily handled the
  team's Track 5 project.

A Devpost-ready project description is available in
[`docs/DEVPOST.md`](docs/DEVPOST.md). The public demo-video URL is still pending there.

Project-authored work is available under the MIT License. The organizer
benchmark, problem-statement restatement, optional dependencies, trademarks,
and future media are scoped in [`NOTICE.md`](NOTICE.md).

Performance results are valid only when baseline and optimized runs use the
same GPU, dtype, official shape, seed, warmup, repeats, compile configuration
and TF32 policy. Historical measurements remain in `SOLUTION.md`; only the
fresh standalone run under `results/final/` is attributed to the submitted
artifact.
