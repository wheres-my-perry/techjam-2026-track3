# Promoted final evidence — driver 595 timeline sweep

These artifacts were produced from commit
`4f77a04fd51c3a2ad8b9d8986657915ae3ca94d6` on 2026-08-31 using the active
`main.py` → `v16_1_clean.py` submission path.

## Environment

- Vast.ai Ubuntu 24.04.4 container, Linux `5.15.0-187-generic`.
- AMD Ryzen 5 5600X, 12 visible logical CPUs, 33,564,246,016 bytes RAM.
- NVIDIA GeForce RTX 5090, compute capability `12.0`, 32,607 MiB visible VRAM.
- NVIDIA driver `595.71.05`.
- Python `3.12.14`.
- PyTorch `2.11.0+cu128`, CUDA wheel `12.8`, cuDNN `9.19.0`.
- Triton `3.6.0`.
- Public tensors and outputs are FP32; eligible internal compute uses FP16.
- TF32 enabled for baseline and optimized paths; matmul precision `high`.

The exact machine-readable inventory is in `environment.json`.

## Official shapes #1–#13

Command:

```bash
CUDA_VISIBLE_DEVICES=0 /venv/main/bin/python timeline_runner.py \
  --python /venv/main/bin/python \
  --checkpoints v16_1,baseline,v1,v2,v3_1_eager,v3_1_compiled,v4_1,v4_2,v4_3,v8,v11,v16_1 \
  --shape-ids 1-13 \
  --device cuda:0 --dtype float32 \
  --accuracy-trials 5 \
  --warmup 20 --repeats 100 --benchmark-rounds 3 \
  --compile-mode max-autotune --timeout 1800 \
  --control-drift-threshold 0.03 --run-id full14-sweep-r1 \
  --environment-id rtx5090-sm120-driver595.71.05-torch2.11-cu128 \
  --source-revision 4f77a04fd51c3a2ad8b9d8986657915ae3ca94d6
```

All 13 shapes passed the strict competition comparator with zero failed
elements. The maximum absolute error across the matrix was `0.00179085`.
The predeclared V16.1 start-control geometric-mean speedup was `11.803x`;
per-shape speedup ranged from `2.379x` to `38.762x`. The end-control measured
`11.838x`. Start/end baseline and optimized geomean drift were `0.166%` and
`0.458%`, so the 3% drift gate passed.

The earlier driver-580 evidence is preserved under
`results/cross-host-driver580/`. Its baseline geomean was `72.48%` faster than
the driver-595 baseline, while its optimized geomean was `15.51%` faster.
Therefore `7.904x → 11.803x` is a cross-host ratio change, not a code gain.

Raw artifacts:

- `main_shapes_1_13.json`
- `main_shapes_1_13.csv`

## Official shape #14

The original baseline cannot be timed because its explicit attention score
would require approximately 18.6 TiB. Shape #14 therefore uses a
memory-bounded reference for correctness and reports optimized-only latency.

Full strict accuracy command:

```bash
CUDA_VISIBLE_DEVICES=0 /venv/main/bin/python shape14_timeline_runner.py \
  --python /venv/main/bin/python --checkpoints baseline,v16_1 \
  --device cuda:0 --seed 1234 --batch-limit 32 \
  --query-chunk 256 --compare-token-chunk 2048 --warmup 1 --repeats 5 \
  --compile-mode max-autotune --b1-timeout 900 --streamed-timeout 3600 \
  --native-timeout 1800 --run-id full14-sweep-r1 \
  --environment-id rtx5090-sm120-driver595.71.05-torch2.11-cu128 \
  --source-revision 4f77a04fd51c3a2ad8b9d8986657915ae3ca94d6
```

Result: PASS, `0/3,276,800,000` failed elements, max absolute error
`0.000944197`, mean absolute error `6.56367e-05`, elapsed `330.783 s`, and
accuracy-window peak allocation `19.967 GiB`.

The full log includes non-fatal TorchInductor autotune messages for candidate
Triton configurations that exceeded the SM120 shared-memory limit. Inductor
discarded those choices, selected valid kernels and completed all 32 batches;
these messages are not a model OOM or an accuracy failure.

The same command isolates B1 accuracy, streamed-B32 accuracy, native-B32 probe
and optimized-only timing in separate child processes.

Native B32 probe also passed the exact FP32 output contract
`[32,100000,1024]`. Result: median `6987.4644 ms`, mean `6989.3152 ms`, p90
`6994.0999 ms`, throughput `457,962.98 token/s`, and peak allocation
`24.487 GiB`. Baseline
latency and speedup are N/A.

Raw artifacts:

- `main_shape14_accuracy.log`
- `main_shape14_benchmark.log`
- `main_shape14_matrix.json`
- `main_shape14_summary.json`
