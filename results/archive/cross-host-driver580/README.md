# Archived driver-580 final evidence

This directory preserves the earlier same-host evidence produced with NVIDIA
driver `580.159.03`. It is retained for cross-host audit and is no longer the
promoted `results/final/` dataset. Do not compare its `7.904x` geomean directly
with driver-595 `11.803x` as a code delta: the measured baseline and optimized
latencies changed by different amounts across hosts.

These artifacts were produced from commit
`4f77a04fd51c3a2ad8b9d8986657915ae3ca94d6` on 2026-08-31 using the active
`main.py` → `v16_1_clean.py` submission path.

## Environment

- Vast.ai Ubuntu 24.04.4 container, Linux `5.15.0-187-generic`.
- NVIDIA GeForce RTX 5090, compute capability `12.0`, 32,607 MiB visible VRAM.
- NVIDIA driver `580.159.03`.
- Python `3.12.14`.
- PyTorch `2.11.0+cu128`, CUDA wheel `12.8`, cuDNN `9.19.0`.
- Triton `3.6.0`.
- Public tensors and outputs are FP32; eligible internal compute uses FP16.
- TF32 enabled for baseline and optimized paths; matmul precision `high`.

The exact machine-readable inventory is in `environment.json`.

## Official shapes #1–#13

Command:

```bash
CUDA_VISIBLE_DEVICES=0 /venv/main/bin/python -m tools.matrix_runner \
  --impl main \
  --shape-ids 1,2,3,4,5,6,7,8,9,10,11,12,13 \
  --device cuda:0 --dtype float32 \
  --accuracy-trials 5 \
  --warmup 20 --repeats 100 --benchmark-rounds 3 \
  --compile-user --compile-mode max-autotune --timeout 1800
```

All 13 shapes passed the strict competition comparator with zero failed
elements. The maximum absolute error across the matrix was `0.00179085`.
Geometric-mean speedup was `7.904x`; per-shape speedup ranged from `2.489x` to
`33.925x`.

Raw artifacts:

- `main_shapes_1_13.json`
- `main_shapes_1_13.csv`

## Official shape #14

The original baseline cannot be timed because its explicit attention score
would require approximately 18.6 TiB. Shape #14 therefore uses a
memory-bounded reference for correctness and reports optimized-only latency.

Full strict accuracy command:

```bash
CUDA_VISIBLE_DEVICES=0 /venv/main/bin/python -m tools.shape14.accuracy \
  --device cuda:0 --impl main --batch-limit 32 \
  --query-chunk 256 --compare-token-chunk 2048 \
  --seed 1234 --compile-mode max-autotune
```

Result: PASS, `0/3,276,800,000` failed elements, max absolute error
`0.000944197`, mean absolute error `6.56367e-05`, elapsed `348.534 s`, and
accuracy-window peak allocation `19.967 GiB`.

The full log includes non-fatal TorchInductor autotune messages for candidate
Triton configurations that exceeded the SM120 shared-memory limit. Inductor
discarded those choices, selected valid kernels and completed all 32 batches;
these messages are not a model OOM or an accuracy failure.

Optimized-only timing command:

```bash
CUDA_VISIBLE_DEVICES=0 /venv/main/bin/python -m tools.shape14.optimized_benchmark \
  --device cuda:0 --impl main --warmup 1 --repeats 5 \
  --compile-mode max-autotune
```

Result: median `7213.5254 ms`, mean `7204.1459 ms`, p90 `7252.4406 ms`,
throughput `443,611.11 token/s`, and peak allocation `24.487 GiB`. Baseline
latency and speedup are N/A.

Raw artifacts:

- `main_shape14_accuracy.log`
- `main_shape14_benchmark.log`
- `main_shape14_summary.json`
