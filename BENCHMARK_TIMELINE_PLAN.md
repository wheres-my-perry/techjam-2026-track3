# Benchmark timeline — full official matrix trên RTX 5090

## 1. Mục tiêu và phạm vi

Benchmark lại cùng một environment cho 11 checkpoint chính:

```text
Baseline → V1 → V2 → V3.1 eager → V3.1 compiled → V4.1 → V4.2
         → V4.3 → V8 → V11 → V16.1
```

- Mỗi checkpoint chạy đủ official shapes **#1–#13**. Không dùng subset làm
  score chính.
- Shape **#14 chỉ giữ hai record theo scope cuối của owner**: Baseline và
  V16.1. Không chạy #14 cho các checkpoint lịch sử khác.
- Correctness dùng strict comparator của cuộc thi và luôn là gate trước timing:
  `relative error < 0.02 OR absolute error < 0.002`.
- Không đổi baseline, official shapes, public FP32 input/output, seed, tolerance,
  state dict hoặc `UserOptimizedTransformer.forward(x, valid_token_mask)`.

## 2. Environment khóa

| Thành phần | Giá trị |
|---|---|
| GPU | NVIDIA GeForce RTX 5090, `sm120`, 32,607 MiB |
| Driver | `595.71.05` |
| CPU / RAM | AMD Ryzen 5 5600X, 12 logical CPUs / 33,564,246,016 bytes |
| Python | `3.12.14` |
| PyTorch / CUDA wheel | `2.11.0+cu128` / `12.8` |
| cuDNN / Triton | `9.19.0` / `3.6.0` |
| Source snapshot | `4f77a04fd51c3a2ad8b9d8986657915ae3ca94d6` |
| Environment ID | `rtx5090-sm120-driver595.71.05-torch2.11-cu128` |

Remote snapshot không mang `.git`; revision được khóa bằng giá trị snapshot đã
khai báo và SHA-256 của runner, implementation cùng dependency chain. Endpoint
và credential không được ghi vào repository.

## 3. Tooling

### `timeline_adapter.py`

- Registry checkpoint và dependency chain.
- Load mỗi archived implementation trong process riêng, thêm
  `archive/versions/` vào import path và inject đúng class vào official harness.
- V1 chỉ cung cấp implementation class; comparator, baseline, weight copy,
  input generation và timing vẫn thuộc harness hiện tại.
- Preflight kiểm tra import, forward signature, strict state-dict load và
  bitwise weight equivalence.
- Manifest lưu resolved class/module, compile policy và SHA-256 source/dependency.

### `timeline_runner.py`

- Default `--shape-ids 1-13`; shape #14 bị từ chối và phải dùng runner riêng.
- Hỗ trợ `--checkpoints`, ranges trong `--shape-ids`, `--list-checkpoints`,
  `--list-shapes`, `--preflight-only` và toàn bộ protocol benchmark.
- Mỗi checkpoint/shape chạy trong subprocess riêng; OOM, timeout hoặc accuracy
  failure không làm mất các row đã hoàn tất.
- Checkpoint lặp được gắn occurrence riêng, nhờ đó V16.1 start/end control không
  overwrite nhau.
- Chỉ sinh `official_full_speedup_geomean` khi checkpoint PASS đủ 13/13.
- Sinh `control_drift.json` cho baseline/optimized geomean và heavy shapes
  #6/#8/#13.

### `shape14_checkpoint_worker.py` và `shape14_timeline_runner.py`

- Bốn stage process-isolated:
  `b1-accuracy → streamed-b32-accuracy → native-b32-probe → native-timing`.
- Baseline được ghi `INFEASIBLE_STATIC` vì explicit score cần khoảng 18.6 TiB.
- V16.1 dùng reusable inner executor, outer full-B32 forward giữ eager chunk loop.
- Timing chỉ chạy sau full streamed strict PASS và native B32 PASS; baseline
  latency/speedup luôn N/A.

## 4. Protocol #1–#13

```text
dtype=float32
seed=1234
accuracy_trials=5
warmup=20
repeats=100
benchmark_rounds=3
TF32=true
matmul_precision=high
compile_mode=max-autotune
timeout=1800 seconds/shape
```

- Baseline, V1, V2 và V3.1 eager chạy eager.
- V3.1 compiled, V4.1, V4.2, V4.3, V8, V11 và V16.1 dùng outer
  `torch.compile(mode="max-autotune")`.
- Accuracy-failed row không có timing và không dùng `--benchmark-on-failure`.

Execution order:

1. V16.1 full start-control.
2. Baseline → V11 theo chronology.
3. V16.1 full end-control.
4. Drift gate 3% trên baseline/optimized geomean và #6/#8/#13.
5. Reverse-order full matrices cho các aggregate chênh dưới 3%:
   `V16.1 → V11 → V8 → V4.3` và `V4.2 → V4.1`.

## 5. Protocol shape #14

Scope cuối chỉ gồm Baseline và V16.1:

| Checkpoint | B1 strict | Streamed B32 strict | Native B32 | Timing |
|---|---|---|---|---|
| Baseline | `INFEASIBLE_STATIC` | Skip | Skip | N/A |
| V16.1 | Query-blocked strict | Full 32 strict | Actual `forward(B=32)` | Optimized-only `1/5` |

Parameters:

```text
seed=1234
query_chunk=256
compare_token_chunk=2048
B1 timeout=900 s
streamed timeout=3600 s
native/timing timeout=1800 s
warmup=1
repeats=5
compile_mode=max-autotune (inner executor)
```

Memory-bounded reference chỉ chứng minh arithmetic correctness; nó không được
dùng làm baseline performance. Native probe gọi nguyên V16.1 forward trên input
`[32,100000,1024]` và kiểm tra output contract/peak allocation.

## 6. Commands đã dùng

Full chronological sweep và controls:

```bash
CUDA_VISIBLE_DEVICES=0 /venv/main/bin/python timeline_runner.py \
  --python /venv/main/bin/python \
  --checkpoints v16_1,baseline,v1,v2,v3_1_eager,v3_1_compiled,v4_1,v4_2,v4_3,v8,v11,v16_1 \
  --shape-ids 1-13 --device cuda:0 --dtype float32 \
  --accuracy-trials 5 --warmup 20 --repeats 100 \
  --benchmark-rounds 3 --seed 1234 --timeout 1800 \
  --matmul-precision high --compile-mode max-autotune \
  --control-drift-threshold 0.03 --run-id full14-sweep-r1 \
  --environment-id rtx5090-sm120-driver595.71.05-torch2.11-cu128 \
  --source-revision 4f77a04fd51c3a2ad8b9d8986657915ae3ca94d6
```

Reverse order:

```bash
CUDA_VISIBLE_DEVICES=0 /venv/main/bin/python timeline_runner.py \
  --python /venv/main/bin/python \
  --checkpoints v16_1,v11,v8,v4_3,v4_2,v4_1 --shape-ids 1-13 \
  --device cuda:0 --dtype float32 --accuracy-trials 5 \
  --warmup 20 --repeats 100 --benchmark-rounds 3 --seed 1234 \
  --timeout 1800 --matmul-precision high --compile-mode max-autotune \
  --run-id full14-reverse-r1 \
  --environment-id rtx5090-sm120-driver595.71.05-torch2.11-cu128 \
  --source-revision 4f77a04fd51c3a2ad8b9d8986657915ae3ca94d6
```

Shape #14:

```bash
CUDA_VISIBLE_DEVICES=0 /venv/main/bin/python shape14_timeline_runner.py \
  --python /venv/main/bin/python --checkpoints baseline,v16_1 \
  --device cuda:0 --seed 1234 --batch-limit 32 \
  --query-chunk 256 --compare-token-chunk 2048 \
  --warmup 1 --repeats 5 --compile-mode max-autotune \
  --b1-timeout 900 --streamed-timeout 3600 --native-timeout 1800 \
  --run-id full14-sweep-r1 \
  --environment-id rtx5090-sm120-driver595.71.05-torch2.11-cu128 \
  --source-revision 4f77a04fd51c3a2ad8b9d8986657915ae3ca94d6
```

## 7. Kết quả thực thi

| Checkpoint | Accuracy #1–#13 | Official geomean | Reverse geomean |
|---|---:|---:|---:|
| Baseline | 13/13 PASS | 1.0011x | — |
| V1 | 13/13 PASS | 1.0763x | — |
| V2 | 13/13 PASS | 1.4353x | — |
| V3.1 eager | 13/13 PASS | 2.1006x | — |
| V3.1 compiled | 0/13; timing skipped | N/A | — |
| V4.1 | 13/13 PASS | 10.1999x | 10.1926x |
| V4.2 | 13/13 PASS | 10.4489x | 10.4349x |
| V4.3 | 13/13 PASS | 11.6755x | 11.6948x |
| V8 | 13/13 PASS | 11.7854x | 11.6580x |
| V11 | 13/13 PASS | 11.7483x | 11.7439x |
| V16.1 start | 13/13 PASS | **11.8030x** | 11.7617x |
| V16.1 end | 13/13 PASS | 11.8383x | — |

V16.1 start/end drift gate PASS: baseline geomean `0.166%`, optimized geomean
`0.458%`; maximum heavy-shape drift là `1.042%` ở optimized #8.

V3.1 compiled fail strict cả 13 shapes, tổng `201,682` failed elements qua năm
trial/shape; không có performance row. Reverse evidence cho thấy V4.2 hơn V4.1
ổn định khoảng `2.38–2.44%`, còn V4.3/V8/V11/V16.1 chỉ khác dưới 1% và một số
comparison đổi dấu theo order, nên không diễn giải thành gain chắc chắn.

Shape #14:

- Baseline: `INFEASIBLE_STATIC`; baseline latency/speedup N/A.
- V16.1 B1: PASS `0/102,400,000`.
- V16.1 streamed B32: PASS `0/3,276,800,000`, max abs `0.000944197`.
- V16.1 native B32: PASS output contract `[32,100000,1024]` FP32.
- Optimized-only median `6987.4644 ms`, mean `6989.3152 ms`, p90
  `6994.0999 ms`, throughput `457,962.98 token/s`, peak `24.487 GiB`.

Headline dùng **V16.1 start-control 11.803x** vì đã predeclared trước sweep và
drift gate PASS. Driver-595 host có baseline geomean chậm hơn driver-580 host
`72.48%`, trong khi optimized geomean chậm hơn `15.51%`; do đó không được gọi
chênh `7.904x → 11.803x` là code improvement.

## 8. Artifacts và acceptance

- Raw ignored:
  `benchmark-results/timeline-rtx5090-driver595/{full14-sweep-r1,full14-reverse-r1,full14-preflight-venv}/`.
- Curated timeline: `results/timeline-rtx5090-driver595/`.
- Promoted final: `results/final/`.
- Driver-580 evidence: `results/cross-host-driver580/`.

Acceptance status:

- [x] 11 checkpoint có accuracy đủ #1–#13; timing hoặc lý do skip rõ ràng.
- [x] Không timing accuracy-failed row.
- [x] V16.1 start/end controls trong drift budget.
- [x] Reverse-order evidence cho aggregate difference dưới 3%.
- [x] Baseline/V16.1 có record #14 đúng scope cuối.
- [x] V16.1 full streamed strict và native B32 PASS trước timing.
- [x] Commands, environment, revision và SHA-256 được lưu trong JSON.
- [x] Raw directory ignored; curated artifacts không chứa endpoint/credential.
- [x] Chạy final `py_compile`, JSON validation, documentation consistency và
  `git diff --check` sau khi đồng bộ report.
