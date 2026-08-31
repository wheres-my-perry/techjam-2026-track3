# Báo cáo benchmark toàn bộ checkpoint trên RTX 5090 driver 595

**Ngày chạy:** 2026-08-31  
**Phạm vi:** 11 checkpoint chính trên official shapes #1–#13; shape #14 chỉ
gồm Baseline và V16.1 theo scope cuối của owner.  
**Source snapshot:** `4f77a04fd51c3a2ad8b9d8986657915ae3ca94d6`  
**Environment ID:** `rtx5090-sm120-driver595.71.05-torch2.11-cu128`

## 1. Kết luận chính

- Toàn bộ 11 checkpoint chính đã được accuracy-check trên đủ official shapes
  #1–#13 trong cùng một environment RTX 5090.
- Baseline, V1, V2, V3.1 eager, V4.1, V4.2, V4.3, V8, V11 và V16.1 đều strict
  PASS 13/13 trước timing.
- V3.1 compiled strict FAIL cả 13/13 shapes, tổng `201,682` failed elements;
  timing bị skip toàn bộ theo correctness gate.
- V16.1 start-control strict PASS 13/13, zero failed elements, official
  geometric-mean speedup **11.802998698x**, báo cáo làm **11.803x**.
- V16.1 end-control đạt `11.838288178x`. Baseline/optimized geomean và ba
  heavy shapes #6/#8/#13 đều nằm trong drift budget 3%; gate PASS.
- Reverse-order sweep xác nhận V4.2 hơn V4.1 ổn định khoảng `2.38–2.44%`.
  Các chênh lệch V4.3/V8/V11/V16.1 dưới 1% và có comparison đổi dấu theo
  order, nên không được diễn giải thành code gain chắc chắn.
- Shape #14 chỉ có hai record:
  - Baseline: `INFEASIBLE_STATIC`, latency và speedup N/A vì explicit score
    tensor cần khoảng `18.6 TiB`.
  - V16.1: B1 strict PASS, full streamed B32 strict PASS
    `0/3,276,800,000`, native B32 PASS, optimized-only median
    `6987.4644 ms`.
- Headline driver-595 không được so trực tiếp thành code improvement từ
  driver-580 `7.904x`: cùng source revision nhưng baseline host mới chậm hơn
  `72.48%`, optimized host mới chậm hơn `15.51%`.

## 2. Mục tiêu, phạm vi và nguyên tắc

### 2.1 Mục tiêu

Đo lại tiến trình tối ưu theo chronology trên cùng GPU/software stack:

```text
Baseline → V1 → V2 → V3.1 eager → V3.1 compiled → V4.1 → V4.2
         → V4.3 → V8 → V11 → V16.1
```

Mỗi checkpoint phải được đánh giá trên đủ #1–#13. Kết quả aggregate chỉ là
official full-matrix score khi checkpoint strict PASS đủ 13/13.

### 2.2 Scope shape #14

Theo quyết định cuối của owner, shape #14 chỉ chạy:

```text
Baseline feasibility record → V16.1 B1 accuracy → V16.1 streamed B32 accuracy
→ V16.1 native B32 probe → V16.1 optimized-only timing
```

Không report #14 cho V1–V11. Những process lịch sử #14 đã bắt đầu trước khi
scope được thu hẹp đã bị dừng; raw và curated matrix cuối chỉ chứa
`baseline,v16_1`.

### 2.3 Correctness gate

Comparator được giữ đúng luật cuộc thi, với hai phép so sánh strict `<`:

```text
absolute_error < 0.002 OR relative_error < 0.02
```

Một row chỉ được timing khi toàn bộ accuracy trials PASS. Không dùng
`--benchmark-on-failure` trong report này.

## 3. Environment thực thi

| Thành phần | Giá trị |
|---|---|
| Provider/container | Vast.ai, Ubuntu 24.04.4 LTS |
| Kernel | Linux `5.15.0-187-generic` |
| GPU | NVIDIA GeForce RTX 5090, `sm120` |
| Visible VRAM | `32,607 MiB` |
| NVIDIA driver | `595.71.05` |
| System CUDA toolkit | `12.8` |
| CPU | AMD Ryzen 5 5600X 6-Core Processor |
| Logical CPU | `12` |
| RAM | `33,564,246,016 bytes` |
| Workspace overlay | `17,179,869,184 bytes` (`16 GiB`) |
| Persistent volume | Không |
| Python | `3.12.14` |
| PyTorch | `2.11.0+cu128` |
| PyTorch CUDA | `12.8` |
| cuDNN | `9.19.0` |
| Triton | `3.6.0` |
| Visible device trong process | `cuda:0` qua `CUDA_VISIBLE_DEVICES=0` |

Tại đầu full sweep, `nvidia-smi` ghi GPU ở `P5`, `35°C`, SM clock `247 MHz`,
power `8.76 W` và không có compute process khác. Đây là trạng thái idle trước
benchmark; clock thấp là idle clock, không phải clock dùng trong timed region.

Remote snapshot không chứa `.git`. Revision được khóa bằng declared snapshot
và SHA-256 của runner, adapter, implementation cùng dependency chain. Endpoint
SSH và credential không được ghi vào artifact hoặc report.

### 3.1 Thời gian các run chính

| Run ID | Nội dung | UTC | Asia/Singapore | Wall time |
|---|---|---|---|---:|
| `full14-preflight-venv` | Preflight 11 checkpoint | 13:31:55–13:32:08 | 21:31:55–21:32:08 | 12.8 s |
| `full14-sweep-r1` | Chronological #1–#13 + controls | 13:32:43–14:09:35 | 21:32:43–22:09:35 | 36 m 52.4 s |
| `full14-reverse-r1` | Reverse-order full matrices | 14:11:42–14:30:08 | 22:11:42–22:30:08 | 18 m 26.4 s |
| `full14-sweep-r1/shape14` | Baseline/V16.1 #14 | 14:35:06–14:42:08 | 22:35:06–22:42:08 | 7 m 1.7 s |

## 4. Source manifest và preflight

### 4.1 Checkpoint resolution

| ID | Resolved label/source | Execution | Implementation SHA-256 | Preflight |
|---|---|---|---|---|
| Baseline | `torch_transformer_benchmark.py` | eager | `c072c48f22cb1438fe903c269eac9039c2554e0c247dcdbe147b9fe950af9500` | PASS |
| V1 | `archive/versions/v1_fuseQKV.py` | eager | `fff5ee507ae778c1f193f58119e807ede0abdb483c0156a7ebb8ed7aee2dcb72` | PASS |
| V2 | `archive/versions/v2_SPDA.py` | eager | `f797f49c152023190d52fa0859904ed394eb5ae8aabd37abb0711a79ba1d5307` | PASS |
| V3.1 eager | `archive/versions/v3_1_CausalMask.py` | eager | `fda24f4a120c955c0dffef838fc303ee8e7262da0e18328f8e274652c84b884b` | PASS |
| V3.1 compiled | `archive/versions/v3_1_CausalMask.py` | outer max-autotune | `fda24f4a120c955c0dffef838fc303ee8e7262da0e18328f8e274652c84b884b` | PASS |
| V4.1 | `archive/versions/v4_1_FP16_GELU.py` | outer max-autotune | `f9d0374ea9fbdd3a905ae3a857e5f90945a387adbcde8f901536e3fe221c3d45` | PASS |
| V4.2 | `archive/versions/v4_2_SDPA_Dispatch.py` | outer max-autotune | `e7ca9dc90500482fb4a1ea7f5815369c38e318c89c2b0e25b553c686e3c05c6e` | PASS |
| V4.3 | `archive/versions/v4_3_Flash.py` | outer max-autotune | `efd02ccb4790ff39c68243312e0512bc6ce834304e264cfa33c5e04472fccafc` | PASS |
| V8 | `archive/versions/v8_FusedFFNGELU.py` | outer max-autotune | `946d316366bae66c5001d4cb8308c8e071d4e8b761576ebfbe894ca586080115` | PASS |
| V11 | `archive/versions/v11_FP32PreGELU.py` | outer max-autotune | `5f0a3ada4a6a75207c867312e5b587dab988bbabf8e887a2a098532105e92aa3` | PASS |
| V16.1 | `v16_1_clean.py` | inner executor; outer #1–#13 max-autotune | `522feff97b482e920d3dde542a659473bdc66ae04757205ab4b9b7c2e209025c` | PASS |

Dependency chung `torch_transformer_benchmark.py` có SHA-256
`c072c48f22cb1438fe903c269eac9039c2554e0c247dcdbe147b9fe950af9500`.
V4.1–V11 dependency `archive/versions/v4_mixed_precision_common.py` có SHA-256
`789823ecf8a33565f72d4aa98c209596b9bb32a931931617cbd0aec65745d8d2`.

Runner hashes dùng trong actual run:

- `timeline_runner.py`:
  `0956cd5dc914b87e3134ab966870adc8912e60e4dd55d7392ae135efad858606`.
- `timeline_adapter.py`:
  `b0ee017fc9f283fdd39535b8518606904be6675bc8819c2d486ea794d644f8a0`.
- `shape14_timeline_runner.py`:
  `d46d2462ae8d845bf96980203623d5ddc9a655e5dbe68056185df5b2820dfa88`.
- `shape14_checkpoint_worker.py`:
  `79260eb4e2b46d4719d2498fbb746fb1a66cb86e34261f157a2f4aa9b4437025`.

### 4.2 Preflight result

Tất cả 11 checkpoint PASS:

- syntax/import và class resolution;
- forward signature `forward(self, x, valid_token_mask)`;
- strict `state_dict` load;
- `66` state-dict keys cho mỗi checkpoint;
- bitwise weight equivalence sau copy/load;
- source và dependency SHA-256 manifest.

Preflight PASS không thay thế accuracy. Mỗi official row vẫn chạy 5 accuracy
trials trước timing.

## 5. Official workload #1–#13

| ID | B | S | D | H | Layers | FFN | Causal |
|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 64 | 128 | 128 | 4 | 4 | 128 | true |
| 2 | 1 | 128 | 128 | 4 | 4 | 128 | true |
| 3 | 4 | 128 | 128 | 4 | 4 | 128 | true |
| 4 | 16 | 128 | 128 | 4 | 4 | 128 | true |
| 5 | 128 | 128 | 128 | 4 | 4 | 128 | true |
| 6 | 10000 | 128 | 128 | 4 | 4 | 128 | true |
| 7 | 64 | 128 | 32 | 4 | 4 | 32 | true |
| 8 | 64 | 128 | 1024 | 4 | 4 | 1024 | true |
| 9 | 64 | 128 | 128 | 1 | 4 | 128 | true |
| 10 | 64 | 128 | 128 | 2 | 4 | 128 | true |
| 11 | 64 | 128 | 128 | 16 | 4 | 128 | true |
| 12 | 64 | 32 | 128 | 4 | 4 | 128 | true |
| 13 | 64 | 1024 | 128 | 4 | 4 | 128 | true |

## 6. Benchmark protocol #1–#13

```text
dtype=float32
seed=1234
accuracy_trials=5
warmup=20
repeats=100
benchmark_rounds=3
TF32=true
matmul_precision=high
timeout=1800 seconds/shape
compile_mode=max-autotune cho compiled checkpoint
```

Timing dùng CUDA Event trên current stream. Random-data generation, compilation
và autotuning không nằm trong steady-state timed window. Mỗi round dùng cùng
input cố định; baseline và optimized đổi order theo round trong official
harness.

Full chronological order:

1. V16.1 start-control.
2. Baseline → V11 theo chronology.
3. V16.1 end-control.
4. Drift gate trên baseline/optimized geomean và #6/#8/#13.
5. Reverse-order full matrices cho checkpoint có aggregate difference dưới 3%.

Official score là geometric mean của 13 per-shape median speedups:

```text
exp(mean(log(baseline_median_i / optimized_median_i)))
```

Score chỉ được công bố khi cả 13 rows strict PASS.

## 7. Exact commands

### 7.1 Chronological sweep và V16.1 controls

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

### 7.2 Reverse-order full matrices

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

### 7.3 Shape #14 Baseline/V16.1

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

## 8. Full chronological results #1–#13

### 8.1 Aggregate summary

`Baseline GM` và `Optimized GM` là geometric mean latency của 13 shapes, không
phải arithmetic mean.

| Run | PASS | Failed | Worst max abs | Baseline GM ms | Optimized GM ms | Speedup GM |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 13/13 | 0 | 0 | 3.473164 | 3.469317 | 1.001076x |
| V1 | 13/13 | 0 | 0.000607941 | 3.485113 | 3.237838 | 1.076311x |
| V2 | 13/13 | 0 | 0.00132750 | 3.484281 | 2.427404 | 1.435338x |
| V3.1 eager | 13/13 | 0 | 0.00132750 | 3.498697 | 1.665486 | 2.100580x |
| V3.1 compiled | 0/13 | 201,682 | 0.00583315 | N/A | N/A | N/A |
| V4.1 | 13/13 | 0 | 0.00188218 | 3.504288 | 0.343553 | 10.199917x |
| V4.2 | 13/13 | 0 | 0.00188218 | 3.494600 | 0.334456 | 10.448914x |
| V4.3 | 13/13 | 0 | 0.00188214 | 3.507082 | 0.300380 | 11.675531x |
| V8 | 13/13 | 0 | 0.00188214 | 3.511732 | 0.297990 | 11.785400x |
| V11 | 13/13 | 0 | 0.00179085 | 3.506250 | 0.298491 | 11.748306x |
| V16.1 start | 13/13 | 0 | 0.00179085 | 3.510795 | 0.297450 | **11.802999x** |
| V16.1 end | 13/13 | 0 | 0.00179085 | 3.504953 | 0.296088 | 11.838288x |

Baseline checkpoint có score xấp xỉ `1.001x`, không đúng tuyệt đối `1.000x`,
vì baseline và injected baseline class vẫn được timing ở hai lượt độc lập và
có measurement noise.

### 8.2 Per-shape speedup matrix

| Run | #1 | #2 | #3 | #4 | #5 | #6 | #7 | #8 | #9 | #10 | #11 | #12 | #13 | GM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 1.000 | 1.002 | 1.001 | 1.001 | 1.002 | 1.000 | 1.004 | 1.000 | 0.999 | 1.002 | 1.001 | 1.002 | 1.000 | 1.0011 |
| V1 | 1.103 | 1.097 | 1.114 | 1.111 | 1.112 | 1.018 | 1.107 | 1.031 | 1.016 | 1.106 | 1.089 | 1.102 | 0.997 | 1.0763 |
| V2 | 1.330 | 1.321 | 1.342 | 1.322 | 1.333 | 1.928 | 1.325 | 1.013 | 1.180 | 1.322 | 1.326 | 1.324 | 3.727 | 1.4353 |
| V3.1 eager | 1.944 | 1.933 | 1.958 | 1.950 | 1.958 | 2.677 | 1.953 | 1.137 | 1.739 | 1.936 | 1.951 | 1.957 | 7.222 | 2.1006 |
| V3.1 compiled | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL | N/A |
| V4.1 | 9.907 | 16.279 | 16.931 | 16.211 | 6.335 | 6.365 | 16.100 | 2.266 | 9.429 | 10.696 | 6.918 | 15.892 | 17.915 | 10.1999 |
| V4.2 | 10.582 | 15.811 | 16.428 | 15.977 | 6.414 | 6.351 | 15.911 | 2.259 | 10.071 | 10.641 | 6.720 | 15.927 | 23.998 | 10.4489 |
| V4.3 | 12.608 | 16.099 | 16.183 | 15.760 | 8.511 | 6.579 | 15.715 | 2.371 | 10.071 | 12.520 | 9.162 | 15.652 | 38.363 | 11.6755 |
| V8 | 12.821 | 15.707 | 16.685 | 16.481 | 8.458 | 7.010 | 15.845 | 2.353 | 10.161 | 12.360 | 9.230 | 15.552 | 38.345 | 11.7854 |
| V11 | 13.114 | 15.800 | 16.456 | 15.053 | 8.045 | 6.989 | 15.793 | 2.365 | 10.379 | 12.742 | 9.392 | 15.610 | 38.864 | 11.7483 |
| V16.1 start | 13.134 | 16.035 | 16.142 | 15.549 | 8.340 | 7.042 | 15.584 | 2.379 | 10.409 | 12.498 | 9.326 | 15.940 | 38.762 | **11.8030** |
| V16.1 end | 13.274 | 15.955 | 16.053 | 16.043 | 8.306 | 7.022 | 15.913 | 2.352 | 10.433 | 12.488 | 9.447 | 15.715 | 38.946 | 11.8383 |

### 8.3 Diễn giải progression

Forward aggregate progression, chỉ nhằm mô tả timeline:

| Transition | Aggregate delta | Diễn giải |
|---|---:|---|
| Baseline → V1 | +7.52% | Fused QKV cho gain nhỏ nhưng V1 #13 còn regression `0.997x`. |
| V1 → V2 | +33.36% | SDPA tạo bước tăng rõ, nhất là #13 `3.727x`. |
| V2 → V3.1 eager | +46.35% | Giảm materialization/data movement và causal-mask overhead. |
| V3.1 eager → V4.1 | +385.58% | Khác cả mixed precision và compiled execution; không phải single-variable attribution. |
| V4.1 → V4.2 | +2.44% | Gain giữ dấu trong reverse order. |
| V4.2 → V4.3 | +11.74% | Flash-first tạo bước tăng lớn và ổn định so với V4.2. |
| V4.3 → V8 | +0.94% | Dưới noise threshold; reverse order đổi thành `-0.31%`. |
| V8 → V11 | -0.31% | Reverse order đổi thành `+0.74%`; không kết luận winner chắc chắn. |
| V11 → V16.1 | +0.47% | Reverse order còn `+0.15%`; V16.1 được giữ vì final architecture/source sạch, không vì gain lớn. |

## 9. V3.1 compiled correctness failure

V3.1 eager PASS đủ 13 shapes, nhưng cùng source được outer compile thì FAIL
strict ở tất cả shapes. Vì vậy đây là evidence cho compile-induced numerical
change, không phải performance regression/speedup evidence.

| Shape | Status | Max abs | Max rel | Failed / total | Timing |
|---:|---|---:|---:|---:|---|
| 1 | ACCURACY_FAIL | 0.00452989 | 112564 | 2,442/5,242,880 | SKIPPED |
| 2 | ACCURACY_FAIL | 0.00368530 | 33.7156 | 27/81,920 | SKIPPED |
| 3 | ACCURACY_FAIL | 0.00385581 | 496.849 | 150/327,680 | SKIPPED |
| 4 | ACCURACY_FAIL | 0.00452989 | 112567 | 587/1,310,720 | SKIPPED |
| 5 | ACCURACY_FAIL | 0.00452989 | 112564 | 4,909/10,485,760 | SKIPPED |
| 6 | ACCURACY_FAIL | 0.00421563 | 5.63661e+08 | 167,787/819,200,000 | SKIPPED |
| 7 | ACCURACY_FAIL | 0.00583315 | 427.625 | 1,268/1,310,720 | SKIPPED |
| 8 | ACCURACY_FAIL | 0.00348020 | 17828.6 | 3,948/41,943,040 | SKIPPED |
| 9 | ACCURACY_FAIL | 0.00452989 | 1439.52 | 2,537/5,242,880 | SKIPPED |
| 10 | ACCURACY_FAIL | 0.00452989 | 3594.06 | 2,474/5,242,880 | SKIPPED |
| 11 | ACCURACY_FAIL | 0.00452989 | 85818.9 | 2,464/5,242,880 | SKIPPED |
| 12 | ACCURACY_FAIL | 0.00452989 | 270.65 | 1,158/1,310,720 | SKIPPED |
| 13 | ACCURACY_FAIL | 0.00437284 | 72089.1 | 11,931/41,943,040 | SKIPPED |

Tổng failed `201,682`; worst max abs `0.00583315` ở #7. `max_rel` có thể rất
lớn ở output gần zero; pass/fail vẫn dùng per-element absolute-OR-relative rule.

## 10. Promoted V16.1 start-control chi tiết

| Shape | B/S/D/H/L/FFN | Baseline median/p90 ms | Optimized median/p90 ms | Speedup | Baseline tok/s | Optimized tok/s | Max abs | Failed |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | `64/128/128/4/4/128` | 1.7640/1.7906 | 0.1343/0.1348 | 13.134x | 4,644,117.26 | 60,995,948.61 | 0.00127273 | 0 |
| 2 | `1/128/128/4/4/128` | 1.7764/1.8114 | 0.1108/0.1159 | 16.035x | 72,055.19 | 1,155,401.48 | 0.00114284 | 0 |
| 3 | `4/128/128/4/4/128` | 1.7882/1.8010 | 0.1108/0.1150 | 16.142x | 286,315.03 | 4,621,605.94 | 0.00124183 | 0 |
| 4 | `16/128/128/4/4/128` | 1.7384/1.7511 | 0.1118/0.1169 | 15.549x | 1,178,062.26 | 18,317,114.73 | 0.00134721 | 0 |
| 5 | `128/128/128/4/4/128` | 1.7535/1.7829 | 0.2102/0.2107 | 8.340x | 9,343,833.00 | 77,929,982.39 | 0.00147235 | 0 |
| 6 | `10000/128/128/4/4/128` | 177.3218/177.3954 | 25.1813/25.3027 | 7.042x | 7,218,515.82 | 50,831,314.74 | 0.00160612 | 0 |
| 7 | `64/128/32/4/4/32` | 1.7424/1.7542 | 0.1118/0.1159 | 15.584x | 4,701,474.73 | 73,268,458.91 | 0.00179085 | 0 |
| 8 | `64/128/1024/4/4/1024` | 6.6464/6.6782 | 2.7936/2.8180 | 2.379x | 1,232,541.02 | 2,932,400.22 | 0.00134873 | 0 |
| 9 | `64/128/128/1/4/128` | 1.5747/1.5973 | 0.1513/0.1532 | 10.409x | 5,202,300.53 | 54,151,242.44 | 0.00126606 | 0 |
| 10 | `64/128/128/2/4/128` | 1.7550/1.7670 | 0.1404/0.1410 | 12.498x | 4,667,870.01 | 58,340,931.34 | 0.00134724 | 0 |
| 11 | `64/128/128/16/4/128` | 1.7345/1.7522 | 0.1860/0.1880 | 9.326x | 4,723,072.98 | 44,046,799.59 | 0.00140083 | 0 |
| 12 | `64/32/128/4/4/128` | 1.7501/1.7624 | 0.1098/0.1159 | 15.940x | 1,170,221.50 | 18,653,453.54 | 0.00134721 | 0 |
| 13 | `64/1024/128/4/4/128` | 41.8362/41.8658 | 1.0793/1.1550 | 38.762x | 1,566,490.52 | 60,721,063.02 | 0.00147235 | 0 |

Summary:

- failed elements: `0`;
- worst max abs: `0.00179085` ở #7;
- baseline latency GM: `3.5107953795 ms`;
- optimized latency GM: `0.2974497927 ms`;
- official speedup GM: `11.8029986983x`;
- lowest per-shape speedup: `2.379x` ở #8;
- highest per-shape speedup: `38.762x` ở #13.

## 11. Start/end control và drift gate

Predeclared threshold là `3%`. Tất cả metrics PASS:

| Metric | Start | End | Relative drift | Gate |
|---|---:|---:|---:|---|
| Baseline latency GM | 3.510795 ms | 3.504953 ms | 0.1664% | PASS |
| Optimized latency GM | 0.297450 ms | 0.296088 ms | 0.4578% | PASS |
| #6 baseline | 177.3218 ms | 177.3779 ms | 0.0316% | PASS |
| #6 optimized | 25.1813 ms | 25.2597 ms | 0.3113% | PASS |
| #8 baseline | 6.6464 ms | 6.6399 ms | 0.0978% | PASS |
| #8 optimized | 2.7936 ms | 2.8227 ms | 1.0417% | PASS |
| #13 baseline | 41.8362 ms | 41.8442 ms | 0.0191% | PASS |
| #13 optimized | 1.0793 ms | 1.0744 ms | 0.4540% | PASS |

Max observed control drift là `1.0417%`, còn dưới 3%. Vì start-control đã được
predeclare làm candidate cho headline, report dùng `11.803x`, không thay bằng
end-control `11.838x` hoặc chọn run đẹp nhất.

## 12. Reverse-order evidence

### 12.1 Aggregate reproducibility

| Checkpoint | Forward GM | Reverse GM | Reverse so với forward |
|---|---:|---:|---:|
| V4.1 | 10.199917x | 10.192561x | -0.072% |
| V4.2 | 10.448914x | 10.434897x | -0.134% |
| V4.3 | 11.675531x | 11.694844x | +0.165% |
| V8 | 11.785400x | 11.658048x | -1.081% |
| V11 | 11.748306x | 11.743902x | -0.037% |
| V16.1 | 11.802999x | 11.761749x | -0.349% |

### 12.2 Paired interpretation

| Comparison | Forward delta | Reverse delta | Kết luận |
|---|---:|---:|---|
| V4.2 vs V4.1 | +2.441% | +2.378% | Giữ dấu, stable small gain. |
| V8 vs V4.3 | +0.941% | -0.315% | Đổi dấu; không claim V8 chắc chắn nhanh hơn. |
| V11 vs V8 | -0.315% | +0.736% | Đổi dấu; coi như cùng performance tier. |
| V16.1 vs V11 | +0.466% | +0.152% | Giữ dấu nhưng rất nhỏ; không phải lý do chính để promote. |

Kết luận thống kê thực dụng: V4.1 → V4.2 là small win có evidence hai orders;
V4.3, V8, V11 và V16.1 nằm trong cùng một performance band khoảng 11.7–11.8x
trên host này. V16.1 vẫn là main vì correctness, standalone source và
shape-#14 executor, không vì aggregate hơn hẳn các checkpoint cuối.

## 13. Shape #14 report

### 13.1 Official configuration và memory feasibility

```text
B=32, S=100000, D=1024, H=16, layers=2, FFN=1024, causal=true
input dtype=float32, output dtype=float32
```

Input và output mỗi tensor chiếm khoảng `12.207 GiB`. Explicit attention score
riêng đã cần:

```text
32 × 16 × 100000 × 100000 × 4 bytes ≈ 18.6 TiB
```

Do đó original Baseline không thể cung cấp latency trên GPU 32 GiB. Report giữ
baseline latency và speedup là N/A; query-blocked/streamed reference chỉ dùng
cho strict correctness, không được thay thế thành performance baseline.

### 13.2 Stage protocol

| Stage | Protocol | Timeout |
|---|---|---:|
| B1 strict | Exact config/seed, query-blocked reference, B=1 | 900 s |
| Streamed B32 strict | `batch_limit=32`, `query_chunk=256`, `compare_token_chunk=2048` | 3600 s |
| Native B32 probe | Actual V16.1 `forward` trên nguyên `[32,100000,1024]` | 1800 s |
| Native timing | warmup 1, repeats 5, CUDA Event | 1800 s |

Mỗi stage chạy process riêng để OOM/error không làm hỏng stage hoặc checkpoint
tiếp theo.

### 13.3 Baseline result

| Stage | Status | Ghi chú |
|---|---|---|
| B1 | `INFEASIBLE_STATIC` | Explicit `S²` score tensor khoảng 18.6 TiB. |
| Streamed B32 | SKIP | Baseline arithmetic đã được memory-bounded reference dùng làm oracle, không chạy native checkpoint. |
| Native B32 | SKIP | Không executable trên 32 GiB. |
| Timing | N/A | Không có paired baseline latency/speedup. |

### 13.4 V16.1 correctness và native feasibility

| Stage | Status | Failed / total | Max abs | Mean abs | Elapsed | Peak allocated |
|---|---|---:|---:|---:|---:|---:|
| B1 strict | PASS | 0/102,400,000 | 0.0006406903 | — | 15.294 s | 8.138 GiB |
| Streamed B32 strict | PASS | 0/3,276,800,000 | 0.0009441972 | 0.0000656367 | 330.783 s | 19.967 GiB |
| Native B32 probe | PASS | output contract PASS | — | — | 11.918 s | 26.030 GiB |

Native output contract:

```text
shape=[32,100000,1024]
dtype=torch.float32
```

Streamed strict `max_rel=151,706,320` xuất hiện ở reference/output gần zero.
Nó không làm row fail vì mọi element vẫn thỏa nhánh absolute hoặc relative của
strict OR comparator.

### 13.5 V16.1 optimized-only timing

```text
samples_ms = [
  6987.46435546875,
  6983.923828125,
  6987.4033203125,
  6992.8544921875,
  6994.93017578125
]
```

| Metric | Giá trị |
|---|---:|
| Median | `6987.4644 ms` |
| Mean | `6989.3152 ms` |
| p90 | `6994.0999 ms` |
| Min | `6983.9238 ms` |
| Throughput | `457,962.98 token/s` |
| Peak allocated | `24.487 GiB` |
| Baseline latency | N/A |
| Speedup | N/A |

Kết luận #14: V16.1 chứng minh cả arithmetic correctness và native B32
scalability trên 32 GiB. Con số `6987.4644 ms` chỉ là optimized-only latency,
không phải paired speedup.

## 14. Cross-host driver-580 versus driver-595

Source revision giống nhau, nhưng environment phần cứng host khác CPU/RAM và
driver. So sánh chỉ dùng để audit host effect:

| Metric | Driver 580 host | Driver 595 host | Driver-595 delta |
|---|---:|---:|---:|
| Baseline latency GM | 2.035476 ms | 3.510795 ms | +72.48% chậm hơn |
| Optimized latency GM | 0.257521 ms | 0.297450 ms | +15.51% chậm hơn |
| Speedup ratio | 7.904124x | 11.802999x | +49.33% ratio |

Speedup là tỷ số của hai latency. Baseline trên host driver-595 chậm đi nhiều
hơn optimized path, nên tỷ số tăng từ `7.904x` lên `11.803x` dù optimized
latency tuyệt đối cũng chậm hơn. Vì vậy:

> `7.904x → 11.803x` là cross-host ratio effect, không phải code improvement.

Historical driver-580 evidence được giữ nguyên trong
`results/cross-host-driver580/`; không trộn row từ hai host vào cùng aggregate.

## 15. Artifact inventory

### 15.1 Raw, ignored

```text
benchmark-results/timeline-rtx5090-driver595/full14-preflight-venv/
benchmark-results/timeline-rtx5090-driver595/full14-sweep-r1/
benchmark-results/timeline-rtx5090-driver595/full14-reverse-r1/
```

### 15.2 Curated timeline

```text
results/timeline-rtx5090-driver595/
├── environment.json
├── run_metadata.json
├── timeline_summary.json
├── control_drift.json
├── preflight/
├── full/
├── reverse/
└── shape14/
```

`full/` chứa JSON/CSV cho 12 occurrence: V16.1 start, 10 checkpoint còn lại và
V16.1 end. `reverse/` chứa V16.1, V11, V8, V4.3, V4.2 và V4.1. `shape14/` chỉ
chứa Baseline/V16.1.

### 15.3 Promoted final

```text
results/final/
├── README.md
├── environment.json
├── main_shapes_1_13.json
├── main_shapes_1_13.csv
├── main_shape14_matrix.json
├── main_shape14_summary.json
├── main_shape14_accuracy.log
└── main_shape14_benchmark.log
```

`main_shapes_1_13.*` là exact V16.1 start-control, không phải end hoặc reverse
run. `main_shape14_matrix.json` là exact Baseline/V16.1 curated matrix.

## 16. Validation sau benchmark

Các gate sau đã PASS:

- `py_compile` cho active code, timeline adapter/runner và shape-#14 worker;
- checkpoint listing và timeline shape listing #1–#13;
- 11/11 import/signature/strict-state-dict/weight-equivalence preflight;
- 12 full matrices có đúng 13 official rows;
- mọi performance row strict PASS trước timing;
- V3.1 compiled accuracy-failed rows không có optimized latency/speedup;
- derived V16.1 start geomean từ raw speedups bằng
  `11.802998698286368`;
- final promoted JSON/CSV khớp exact start-control;
- start/end drift report PASS;
- shape #14 matrix có đúng `{baseline, v16_1}`;
- V16.1 #14 streamed correctness và native B32 PASS trước timing;
- 43 JSON và 32 CSV curated files parse được;
- 244 SHA-256 fields và 456 command/invocation fields được validate;
- curated artifact scan không thấy SSH endpoint, credential hoặc private key;
- historical ad-hoc remote paths được `.gitignore` chặn;
- `git diff --check` PASS.

## 17. Hạn chế và cách sử dụng kết quả

- Kết quả đại diện cho một RTX 5090/SM120 software stack cụ thể; chưa chứng minh
  portability sang GPU/driver khác.
- Accuracy official dùng seed `1234`. Nhiều seed, input scale và padding ratio
  hơn sẽ tăng robustness nhưng không thuộc sweep này.
- Compile/autotune cold-start bị loại khỏi steady-state timing; report không
  tuyên bố startup latency.
- Difference dưới 3% cần được xem cùng reverse/sandwich evidence. Không dùng
  single forward order để promote V8/V11/V16.1 theo performance.
- Shape #14 không có executable original baseline, nên latency/speedup luôn N/A.
- Peak memory #14 là PyTorch peak allocated, không phải toàn bộ process/GPU
  resident memory.
- V16.1 được promote làm main vì tổng hợp correctness, standalone packaging,
  full-matrix performance và native shape-#14 feasibility. Timeline không tự
  động promote implementation cũ hoặc experimental V19/V19.1.

## 18. Final reporting decision

Submission headline được khóa như sau:

```text
V16.1 standalone final
Official shapes #1–#13: strict PASS 13/13, failed=0
Geometric-mean speedup: 11.803x (predeclared start-control)
Worst max abs: 0.00179085

Official shape #14:
streamed strict PASS 0/3,276,800,000
native B32 PASS
optimized-only median 6987.4644 ms
baseline latency/speedup N/A
```

V16.1 end-control `11.838x` và reverse `11.762x` được giữ để chứng minh drift
và order sensitivity, không thay thế headline. Driver-580 `7.904x` được giữ
làm cross-host archive, không diễn giải thành code delta.

## 19. Nguồn dữ liệu chính

- `results/timeline-rtx5090-driver595/timeline_summary.json`
- `results/timeline-rtx5090-driver595/control_drift.json`
- `results/timeline-rtx5090-driver595/full/`
- `results/timeline-rtx5090-driver595/reverse/`
- `results/timeline-rtx5090-driver595/shape14/shape14_matrix.json`
- `results/final/main_shapes_1_13.json`
- `results/final/main_shape14_summary.json`
- `results/final/environment.json`

Report này chỉ tổng hợp measured evidence từ các artifact trên; không thêm
latency ước lượng hoặc chọn lại run sau khi xem kết quả.
