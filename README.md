# TikTok TechJam 2026 — Track 3 Transformer GPU Optimization

PyTorch implementation of a GPU-optimized Transformer layer under the contest
correctness rule:

```text
relative_error < 0.02 OR absolute_error < 0.002
```

Both comparisons are strict and correctness is checked before performance.

## Active implementation

The repository root now contains one versioned implementation:

- `v16_1_clean.py` — standalone V16.1 model, config, FP16 inference cache,
  Flash-first attention, Triton FP32-pre-GELU kernel and memory-bounded compiled
  executor for long sequences.
- `main.py` — thin adapter that connects the standalone class to the official
  benchmark CLI.

`v16_1_clean.py` imports no benchmark code and no earlier implementation file.
Its only runtime dependencies are PyTorch and optional Triton. Historical
`v1`–`v18` files are preserved under `archive/versions/`; they are not active
runner aliases. Their results and decisions remain in `EXPERIMENTS.md`,
`SOLUTION.md` and `DECISION.md`.

The active algorithm keeps LayerNorm, residuals and public output in FP32;
uses FP16 operands for QKV, SDPA, projections and FFN GEMMs; evaluates exact
GELU from the FP32 FFN-in accumulator; and uses a reusable compiled B=1 body
inside an eager batch loop when `B > 1` and `S >= 8192`.

## Quick start

Create an environment with a CUDA-enabled PyTorch build compatible with the
target GPU. Triton is normally included with the CUDA PyTorch wheel.

List the 14 official shapes:

```bash
python3 matrix_runner.py --list-shapes
```

Run main on official shape #1:

```bash
CUDA_VISIBLE_DEVICES=1 python3 matrix_runner.py \
  --impl main --shape-ids 1 \
  --device cuda:0 --dtype float32 \
  --compile-user --compile-mode max-autotune
```

Run official shapes #1–#13:

```bash
CUDA_VISIBLE_DEVICES=1 python3 matrix_runner.py \
  --impl main \
  --shape-ids 1,2,3,4,5,6,7,8,9,10,11,12,13 \
  --device cuda:0 --dtype float32 \
  --accuracy-trials 5 \
  --compile-user --compile-mode max-autotune
```

The benchmark machine assigns this project physical GPU index `1`. After
`CUDA_VISIBLE_DEVICES=1`, PyTorch correctly addresses it as `cuda:0`.

Official shape #14 requires the memory-bounded accuracy and optimized-only
tools because the original reference would materialize an approximately
18.6 TiB attention-score tensor:

```bash
CUDA_VISIBLE_DEVICES=1 python3 shape14_accuracy.py \
  --device cuda:0 --impl main --batch-limit 32 --query-chunk 256 \
  --compare-token-chunk 2048 --compile-mode max-autotune

CUDA_VISIBLE_DEVICES=1 python3 shape14_optimized_benchmark.py \
  --device cuda:0 --impl main --warmup 1 --repeats 5 \
  --compile-mode max-autotune
```

The second command reports optimized-only latency. Baseline latency and speedup
for shape #14 remain N/A.

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

## Validation

Run the active syntax and orchestration checks:

```bash
python3 -m py_compile main.py matrix_runner.py profile_models.py \
  torch_transformer_benchmark.py v16_1_clean.py shape14_accuracy.py \
  shape14_optimized_benchmark.py shape14_profile.py \
  shape14_fa4_probe.py shape14_sage_probe.py
python3 matrix_runner.py --list-shapes
python3 profile_models.py --list-shapes
```

The standalone cleanup has passed local strict state-dict checks, bitwise
causal/non-causal × mask/no-mask equivalence with the composed V16.1, training
and BF16 fallback equivalence, plus compiled-executor reuse/invalidation checks.
A fresh CUDA official matrix is still required before final submission; the
cleanup itself does not create a new speedup claim.

## Repository map

| Path | Role |
|---|---|
| `v16_1_clean.py` | Only active optimized implementation |
| `main.py` | Official single-shape benchmark adapter |
| `torch_transformer_benchmark.py` | Baseline/reference oracle |
| `matrix_runner.py` | Isolated runner for the 14 official shapes |
| `profile_models.py` | Accuracy, timing and profiler runner |
| `shape14_accuracy.py` | Memory-bounded strict accuracy for shape #14 |
| `shape14_optimized_benchmark.py` | Optimized-only shape-#14 timing |
| `archive/versions/` | Historical implementation and opcheck files |
| `STATEMENT.md` | Competition statement and official shapes |
| `ARCHITECTURE.md` | Runtime and repository architecture |
| `EXPERIMENTS.md` | Commands and measured experiment log |
| `SOLUTION.md` | Full technical report |
| `DECISION.md` | Long-term technical decisions |
| `IMPLEMENTATION_PLAN.md` | Current phase and remaining work |

Performance results are valid only when baseline and optimized runs use the
same GPU, dtype, official shape, seed, warmup, repeats, compile configuration
and TF32 policy. Historical measurements are documented in `SOLUTION.md` and
must not be attributed to the new standalone file without a fresh GPU rerun.
