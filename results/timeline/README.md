# RTX 5090 driver-595 checkpoint timeline

This directory contains curated evidence for the same-environment historical
checkpoint sweep executed on 2026-08-31. The source snapshot was
`4f77a04fd51c3a2ad8b9d8986657915ae3ca94d6`; every checkpoint manifest records
SHA-256 hashes for its implementation and dependency chain.

## Environment and protocol

- RTX 5090 `sm120`, 32,607 MiB; NVIDIA driver `595.71.05`.
- Python `3.12.14`; PyTorch `2.11.0+cu128`; CUDA `12.8`; cuDNN `9.19.0`;
  Triton `3.6.0`.
- Shapes #1–#13: FP32, seed 1234, strict accuracy `5` trials, warmup `20`,
  repeats `100`, rounds `3`, TF32 enabled, matmul precision `high`.
- Compiled checkpoints use `max-autotune`; accuracy-failed rows have no timing.

## Full #1–#13 results

| Checkpoint | Strict accuracy | Forward geomean | Reverse geomean |
|---|---:|---:|---:|
| Baseline | 13/13 PASS | 1.0011x | — |
| V1 | 13/13 PASS | 1.0763x | — |
| V2 | 13/13 PASS | 1.4353x | — |
| V3.1 eager | 13/13 PASS | 2.1006x | — |
| V3.1 compiled | 0/13 | N/A | — |
| V4.1 | 13/13 PASS | 10.1999x | 10.1926x |
| V4.2 | 13/13 PASS | 10.4489x | 10.4349x |
| V4.3 | 13/13 PASS | 11.6755x | 11.6948x |
| V8 | 13/13 PASS | 11.7854x | 11.6580x |
| V11 | 13/13 PASS | 11.7483x | 11.7439x |
| V16.1 start | 13/13 PASS | **11.8030x** | 11.7617x |
| V16.1 end | 13/13 PASS | 11.8383x | — |

V3.1 compiled failed strict accuracy on all 13 shapes and therefore has no
performance timings. V16.1 start/end drift passed the 3% budget: baseline and
optimized geomean drift were `0.166%` and `0.458%`; every #6/#8/#13 latency
drift also passed.

The headline is the predeclared V16.1 start-control, not the highest observed
run. Compared with the archived driver-580 host, the driver-595 host's baseline
geomean was `72.48%` slower while optimized geomean was `15.51%` slower. The
change from `7.904x` to `11.803x` is therefore not a code-improvement claim.

## Shape #14

Per the final owner scope, #14 contains only Baseline and V16.1:

- Baseline: `INFEASIBLE_STATIC`; latency and speedup N/A.
- V16.1 B1 strict: PASS `0/102,400,000`.
- V16.1 streamed B32 strict: PASS `0/3,276,800,000`, max abs `0.000944197`.
- V16.1 native B32: PASS, FP32 output `[32,100000,1024]`.
- Optimized-only median `6987.4644 ms`, mean `6989.3152 ms`, p90
  `6994.0999 ms`, throughput `457,962.98 token/s`, peak `24.487 GiB`.

## Layout

- `environment.json`: promoted machine/runtime manifest.
- `run_metadata.json`, `timeline_summary.json`, `control_drift.json`: sweep-level
  metadata and derived reports.
- `full/`: chronological full matrices, including V16.1 start/end.
- `reverse/`: reverse-order full matrices for sub-3% aggregate comparisons.
- `shape14/`: Baseline/V16.1 feasibility, correctness, native probe and timing.
- `preflight/`: import/signature/state-dict/weight-equivalence evidence.

Raw incremental artifacts remain ignored under
`runs/benchmarks/timeline/`.
