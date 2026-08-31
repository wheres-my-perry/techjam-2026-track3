# Nhật ký phương án và thử nghiệm

## 1. Cách dùng tài liệu này

Mỗi phương án cần ghi rõ giả thuyết, thay đổi, coverage correctness, command benchmark, môi trường và kết quả. Chỉ điền speedup sau khi có log đo thực tế.

Technical report tổng hợp cho các implementation hiện tại nằm tại [SOLUTION.md](SOLUTION.md). `EXPERIMENTS.md` giữ vai trò experiment log; khi thuật toán hoặc candidate thay đổi, phải cập nhật cả hai file.

Public repository: [wheres-my-perry/techjam-2026-track3](https://github.com/wheres-my-perry/techjam-2026-track3).

Correctness gate mặc định: `relative error < 0.02 OR absolute error < 0.002` cho từng phần tử. Hai phép so sánh đều là strict `<`.

Performance policy: chỉ benchmark chính thức trên đúng 14 test shapes trong Appendix của `STATEMENT.md`. Shape khác vẫn được giữ trong experiment log khi hữu ích, nhưng phải gắn nhãn **non-official diagnostic** và không được dùng làm headline result.

Trạng thái dùng trong tài liệu:

- **Idea**: chưa triển khai.
- **Implemented**: đã có code nhưng chưa đủ kết quả xác nhận.
- **Validated**: accuracy matrix đã pass trên môi trường ghi kèm.
- **Benchmarked**: đã có số đo hiệu năng tái lập được.
- **Rejected**: không đạt correctness, hiệu năng hoặc chi phí triển khai.

### Taxonomy implementation

- `v1_fuseQKV.py`: chỉ fuse ba projection Q/K/V.
- `v2_SPDA.py`: lấy v1 làm nền và chỉ thay attention core bằng SDPA; vẫn giữ Q/K/V head copies và module dispatch để đo riêng tác động của SDPA.
- `v3_SDPA_NoCopy.py`: cộng thêm packed-QKV views không-copy, mask reuse và flattened whole-model loop.
- `v3_1_CausalMask.py`: giữ v3 và loại causal-mask materialization cùng một lần zero-out padding dư trong mỗi layer.
- V3.1 + `torch.compile`: execution configuration trên cùng implementation/weights, dùng để ablate Inductor mode mà không tạo version file giả.
- `v4_1_FP16_GELU.py`: FP16 internal GEMM/SDPA/GELU, FP32 norm/residual/output.
- `v4_2_SDPA_Dispatch.py`: giữ V4.1 graph và chọn cuDNN SDPA cho các RTX 5090 shapes đã validate, automatic fallback cho shape khác.
- `v4_3_Flash.py`: candidate V4.3 cuối; causal/right-padding bỏ key mask, ưu tiên Flash với cuDNN/Efficient/Math fallback.
- `v4_3_flash_clean.py`: standalone packaging của V4.3 Flash-first; không import benchmark harness và không phải runner target.
- `v4_3_SDPA_CausalFlash_Dispatch.py`: historical static-dispatch ablation, superseded bởi `v4_3_Flash.py`.
- `v5_FP8.py`: per-tensor E4M3 Linear + FP16 SDPA; negative accuracy ablation.
- `v5_1_MXFP8.py` / `v5_2_MXFP8_*.py`: Blackwell MXFP8 full/single-scope
  negative ablations; kernel sanity PASS nhưng model accuracy FAIL.
- `v6_ApproxGELU.py`: V4.3 graph với FP16 GELU dùng tanh approximation thay
  cho exact/erf; candidate phải qua strict accuracy trước performance.
- `v7_ResidualLayerNorm.py`: V4.3 với pre-normalization được pipeline qua
  residual boundary để giữ branch FP16 và expose add/mask/LayerNorm/cast cho
  một fused Inductor kernel; custom Triton chỉ được viết nếu profile còn gap.
- `v8_FusedFFNGELU.py`: V4.3 với custom Triton FP16 FFN-in GEMM + bias + exact
  GELU epilogue; FFN-out và phần còn lại giữ nguyên để cô lập fusion.
- `v8_1_FusedFFNGELUAll.py`: ablation ép V8 custom FFN/GELU trên mọi shape để
  đo trực tiếp chi phí/lợi ích của bỏ static dispatcher; không thay V8 stable.
- `v9_PersistentMLP.py`: ablation fuse cả hai FFN GEMM cùng exact GELU, không
  materialize hidden activation; support envelope ban đầu D/FFN tối đa 128.
- `v10_PersistentFFNIn.py`: ablation giữ FFN-out CUTLASS của V8 nhưng đổi
  FFN-in/exact-GELU sang persistent-CTA scheduling cho large-M D=FFN=128.
- `v11_FP32PreGELU.py`: promoted arithmetic path/rollback; exact GELU đọc trực
  tiếp FP32 FFN-in accumulator rồi vẫn store GELU FP16 cho FFN-out.
- `v12_FP32FFNOut.py`: V11 ablation cho FFN-out GEMM store trực tiếp FP32;
  attention output projection giữ nguyên để cô lập một precision boundary.
- `v12_1_FP32OutProj.py`: V11 ablation chỉ đổi attention out-projection output.
- `v12_2_FP32ResidualOutputs.py`: kết hợp hai FP32 residual-boundary outputs.
- `v13_INT8FFNProbe.py`: accuracy-only ablation cho FFN-in INT8; weight dùng
  symmetric per-output-channel scale, activation dùng dynamic per-token scale,
  dequantize accumulator trước exact GELU và giữ phần còn lại theo V11.

Taxonomy này giữ mỗi bước tối ưu có một ablation độc lập; v1/v2/v3/v3.1 giải
thích nguồn speedup lịch sử, còn active final artifact là `v16_1_clean.py`.

## 2. S0 — Baseline PyTorch

**Trạng thái:** Official baseline imported and smoke benchmarked.

**File:** `torch_transformer_benchmark.py`.

Baseline dùng các PyTorch operator tách rời:

- Ba Linear projection cho Q, K, V.
- Matmul cho attention score.
- FP32 softmax rồi cast về input dtype.
- Matmul với V và output projection.
- Hai LayerNorm và FFN Linear → GELU → Linear cho mỗi layer.

Baseline là oracle correctness và mốc latency. Không tối ưu hoặc sửa baseline để cải thiện con số so sánh.

Attachment cập nhật ngày 2026-08-27 có SHA-256 `5529c96a80799b51f68092e1444a30b17994554dffdf52da98ba701489a7f36e`. Repository giữ cấu trúc của attachment nhưng chuẩn hóa comparator theo trang đề: strict `<`, CLI mặc định `rtol=0.02`, `atol=0.002`.

## 3. S1 — Packed QKV projection

**Trạng thái:** Correctness validated; chỉ có non-official diagnostic timing, official-shape benchmark pending.

**File:** `v1_fuseQKV.py`.

### Giả thuyết

Ba Q/K/V projection riêng tạo thêm kernel launch và đọc input lặp lại. Packed projection thay ba GEMM bằng một GEMM.

### Thay đổi

- Concatenate Q/K/V weights và biases thành cache packed.
- Một `F.linear` sinh QKV rồi tách thành ba tensor.
- Refresh packed QKV sau khi load weights.
- Giữ nguyên attention math và dùng projection gốc khi training.

### Đánh đổi

- Packed weights chỉ hợp lệ cho inference weights cố định; training fallback tránh cache stale.
- Hiệu quả phụ thuộc GPU, dtype và shape.
- Các phần attention, mask, softmax, output projection và FFN chưa được tối ưu.

### Coverage cần xác nhận

| Nhánh | Accuracy | Performance |
|---|---|---|
| FP32, non-causal, không padding | PASS GPU, exact | 1.035x non-official diagnostic |
| FP32, causal, có padding | PASS local, exact | Chưa đo GPU |
| BF16, non-causal, có padding | PASS local, exact | Chưa đo GPU |
| FP16 | Chưa kiểm tra | Chưa đo GPU |

### Command mẫu

```bash
CUDA_VISIBLE_DEVICES=1 python3 v1_fuseQKV.py \
  --device cuda:0 \
  --dtype float32 \
  --batch-size 64 \
  --seq-len 128 \
  --d-model 128 \
  --heads 4 \
  --ffn-dim 128 \
  --layers 4 \
  --causal \
  --accuracy-trials 20 \
  --warmup 50 \
  --repeats 200 \
  --benchmark-rounds 5
```

### Kết quả

Local correctness smoke ngày 2026-08-27 đã pass. Đây không phải kết quả hiệu năng để nộp:

```text
PyTorch: 2.12.1, CPU
FP32 non-causal: PASS, max_abs=0, failed=0/3072
FP32 causal + padding: PASS, max_abs=0, failed=0/3072
BF16 + padding: PASS, max_abs=0, failed=0/256
```

GPU smoke benchmark lịch sử trên default shape, nay chỉ được xem là **non-official diagnostic**:

```bash
cd /home/chim/techjam-2026-track3
source .venv/bin/activate
CUDA_VISIBLE_DEVICES=1 python v1_fuseQKV.py --device cuda:0 --dtype float32
```

Môi trường: GPU vật lý index `1`, NVIDIA GeForce RTX 5090; driver `595.58.03`; PyTorch `2.13.0+cu130`; FP32; TF32 bật; shape `(B=8, S=128, D=512, H=8, FFN=2048, L=6)`; non-causal; không padding; seed `1234`; warmup `20`; repeats `100`; rounds `3`.

| GPU | PyTorch/CUDA | Shape | Dtype/mask | Baseline median | Optimized median | Speedup | Accuracy |
|---|---|---|---|---:|---:|---:|---|
| RTX 5090, physical index 1 | 2.13.0+cu130 / 13.0 | 8×128×512, H=8, FFN=2048, L=6 | FP32, non-causal, no padding | 1.3569 ms | 1.3112 ms | 1.035x | PASS, max_abs=0, max_rel=0 |

Đây không phải test shape chính thức và không được dùng làm kết quả để nộp.

## 4. S2 — V1 + PyTorch SDPA

**Trạng thái:** GPU core shapes benchmarked; full official-shape matrix pending.

**File:** `v2_SPDA.py`.

### Giả thuyết

Giữ nguyên v1 và chỉ thay explicit attention scores/softmax/context bằng PyTorch SDPA để đo riêng lợi ích của attention backend.

### Phạm vi

- Giữ packed QKV `F.linear` và cache lifecycle của v1.
- Giữ `chunk → _split_heads → contiguous` cho Q/K/V.
- Giữ attention module dispatch ở từng layer.
- FP32 dùng `F.scaled_dot_product_attention`.
- Causal không padding dùng `is_causal=True`; causal + padding dùng boolean mask.
- FP16/BF16 giữ attention math của v1; training dùng ba projection gốc.

V2 cố ý chưa có no-copy Q/K/V views, mask reuse ngoài layer loop hoặc flattened forward. Những thay đổi đó thuộc v3.

### Correctness

Local PyTorch `2.12.1` PASS FP32 cho causal/non-causal × padding/no-padding; BF16 causal + padding PASS exact trên reference-math path. GPU RTX 5090 cũng PASS default non-causal và official shape #1 causal với strict tolerance.

### Benchmark lịch sử của cùng đường thuật toán

Môi trường: RTX 5090 vật lý index `1`, PyTorch `2.13.0+cu130`, FP32, TF32 bật, seed `1234`, warmup `20`, repeats `100`, rounds `3`.

```bash
CUDA_VISIBLE_DEVICES=1 python v2_SPDA.py --device cuda:0 --dtype float32
CUDA_VISIBLE_DEVICES=1 python v2_SPDA.py \
  --device cuda:0 --dtype float32 \
  --batch-size 64 --seq-len 128 --d-model 128 \
  --heads 4 --ffn-dim 128 --layers 4 --causal
```

| Config | Accuracy | Baseline median | V2 median | Speedup |
|---|---|---:|---:|---:|
| B8/S128/D512/H8/FFN2048/L6, non-causal (**diagnostic**) | PASS, max_abs=0.000702024 | 1.3767 ms | 0.9951 ms | 1.384x |
| B64/S128/D128/H4/FFN128/L4, causal | PASS, max_abs=0.00105309 | 1.0120 ms | 0.7469 ms | **1.355x** |

Revalidation sau khi tách file:

```bash
CUDA_VISIBLE_DEVICES=1 python v2_SPDA.py \
  --device cuda:0 --dtype float32 --accuracy-trials 1 \
  --warmup 50 --repeats 200 --benchmark-rounds 5
CUDA_VISIBLE_DEVICES=1 python v2_SPDA.py \
  --device cuda:0 --dtype float32 \
  --batch-size 64 --seq-len 128 --d-model 128 \
  --heads 4 --ffn-dim 128 --layers 4 --causal \
  --accuracy-trials 1 --warmup 50 --repeats 200 --benchmark-rounds 5
```

| Config | Accuracy | Baseline median / p90 | V2 median / p90 | Speedup |
|---|---|---:|---:|---:|
| Default non-causal (**diagnostic**) | PASS, max_abs=0.000662863 | 1.6929 / 2.7341 ms | 1.2622 / 2.1748 ms | 1.341x |
| Official shape #1 causal | PASS, max_abs=0.000944674 | 1.0704 / 1.2637 ms | 0.7977 / 0.9665 ms | 1.342x |

Lượt default có p90 variance cao nên chỉ được xem là diagnostic và không dùng để thay bảng benchmark lịch sử.

## 5. S3 — Packed QKV no-copy + SDPA + flattened loop

**Trạng thái:** GPU core branches benchmarked; full official-shape matrix pending.

**File:** `v3_SDPA_NoCopy.py`.

### Giả thuyết

Sau SDPA, chi phí còn lại đến từ ba explicit Q/K/V layout copies, mask allocation lặp lại và Python/module dispatch giữa các layer. Loại các chi phí này sẽ cải thiện latency mà không đổi phép toán model.

### Phạm vi

- Gộp Q/K/V bằng một `F.linear`.
- Dùng `reshape → permute → unbind` để Q/K/V là view, không gọi `.contiguous()` ba lần.
- Tạo/reuse mask một lần ngoài layer loop.
- Inline attention, residual và FFN trong whole-model forward.
- Dùng SDPA cho FP32 causal và non-causal.
- Refresh packed QKV sau `load_state_dict()`.
- Training và FP16/BF16 fallback toàn bộ về reference.

### Ablation

`v1_old.py` từng được phục dựng tạm thời để kiểm tra whole-model loop rồi đã bị xóa sau khi logic tốt hơn được merge vào v3. Phép đo cho thấy chỉ inline block hoặc đổi scale/reshape vẫn quanh `1.53x`; inline toàn bộ model đạt khoảng `1.79x`.

| Candidate/config | Accuracy | Baseline median | Optimized median | Speedup |
|---|---|---:|---:|---:|
| V2, default non-causal (**diagnostic**) | PASS | 1.3767 ms | 0.9951 ms | 1.384x |
| V3 intermediate, default non-causal (**diagnostic**) | PASS | 1.3747 ms | 0.8993 ms | 1.529x |
| V3 final, default non-causal (**diagnostic**) | PASS, max_abs=0.000662863 | 1.3758 ms | 0.7724 ms | 1.781x |
| V2, official shape #1 causal | PASS | 1.0120 ms | 0.7469 ms | 1.355x |
| V3 intermediate: no-copy, còn module dispatch | PASS | 1.0282 ms | 0.6874 ms | 1.496x |
| V3 final: no-copy + flattened loop | PASS, max_abs=0.00105309 | 1.0558 ms | 0.5494 ms | **1.922x** |

Flatten giảm optimized median thêm `14.1%` ở default non-causal và `20.1%` ở official causal so với V3 intermediate.

### Correctness

PyTorch `2.12.1` local:

| Dtype | Causal | Padding | Kết quả |
|---|:---:|:---:|---|
| FP32 | off/on | off/on | PASS |
| BF16 | on | on | PASS, full reference fallback |
| FP16 | on | on | PASS, full reference fallback |

GPU validation bổ sung PASS cho causal + padding (`max_abs=0.00105309`) và non-causal + padding (`max_abs=0.000714183`); BF16/FP16 fallback khớp exact.

### Command benchmark

```bash
cd /home/chim/techjam-2026-track3
source .venv/bin/activate
CUDA_VISIBLE_DEVICES=1 python v3_SDPA_NoCopy.py \
  --device cuda:0 --dtype float32 \
  --batch-size 64 --seq-len 128 --d-model 128 \
  --heads 4 --ffn-dim 128 --layers 4 --causal \
  --accuracy-trials 3 --warmup 20 --repeats 100 --benchmark-rounds 3
```

Môi trường: RTX 5090 vật lý index `1`; PyTorch `2.13.0+cu130`; TF32 bật; seed `1234`. Official shape #1 đạt `1.922x`. Số `1.781x` thuộc default non-causal diagnostic và không còn là headline result. Lượt đo có một process khác giữ khoảng 3.9 GB VRAM nhưng báo 0% utilization; baseline và optimized vẫn chạy trong cùng process. Nên chạy lại full matrix trên GPU hoàn toàn idle trước submission.

## 6. S3.1 — Causal SDPA không materialize mask

**Trạng thái:** GPU core branches validated; official shapes #1 và #13 benchmarked, full matrix pending.

**File:** `v3_1_CausalMask.py`.

### Giả thuyết

V3 kết hợp key-padding mask `[B,1,1,S]` với causal mask `[S,S]`, tạo tensor `[B,1,S,S]` dù SDPA đã nhận `is_causal`. Truyền key mask trực tiếp cùng `is_causal=True` loại allocation theo `B×S²`. Zero-out attention output cũng dư vì invalid query được zero ở cuối cùng block trước khi sang layer tiếp theo.

### Phạm vi

- Giữ nguyên packed QKV no-copy, SDPA và flattened loop của v3.
- Bỏ `_causal_mask` và phép `mask & causal_mask`.
- Truyền key-padding mask trực tiếp cho SDPA với `is_causal=config.causal`.
- Bỏ `masked_fill` trong `_attention`; vẫn zero invalid query một lần cuối mỗi block và sau final LayerNorm.
- Giữ nguyên FP32 inference gate, fallback training/FP16/BF16 và packed-cache lifecycle.

Không thay GELU, compile mode, dtype path hoặc benchmark harness để cô lập đúng hai thay đổi.

### Correctness

- Local PyTorch `2.12.1`: FP32 causal/non-causal × padding/no-padding PASS; BF16 fallback khớp exact.
- RTX 5090: FP32 causal và non-causal với `padding_ratio=0.25` PASS 5/5 seed; max absolute error lần lượt `0.00105309` và `0.000709176`.
- FP16/BF16 causal + padding PASS exact trên full-reference fallback.
- Official shapes #1 và #13 PASS 5/5 trial, `failed=0`; shape #6 batch 10,000 PASS một-trial correctness/OOM smoke.

### Official benchmark và ablation

```bash
CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m tools.matrix_runner \
  --impl v3.1 --shape-ids 1,13 --device cuda:0 --dtype float32
CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m tools.matrix_runner \
  --impl v3 --shape-ids 1,13 --device cuda:0 --dtype float32
```

Môi trường: RTX 5090 vật lý index `1`, PyTorch `2.13.0+cu130`, CUDA `13.0`, TF32 bật, seed `1234`, accuracy trials `5`, warmup `20`, repeats `100`, rounds `3`, không compile.

| Shape | Impl | Accuracy | Baseline median / p90 | Optimized median / p90 | Throughput | Speedup |
|---|---|---|---:|---:|---:|---:|
| #1: B64/S128/D128/H4/L4/FFN128 | V3 | PASS, max_abs=0.00105309 | 1.0252 / 1.0370 ms | 0.5322 / 0.5436 ms | 15,392,478 token/s | 1.926x |
| #1: B64/S128/D128/H4/L4/FFN128 | V3.1 | PASS, max_abs=0.00105309 | 1.0021 / 1.0168 ms | 0.4656 / 0.4777 ms | 17,593,293 token/s | **2.152x** |
| #13: B64/S1024/D128/H4/L4/FFN128 | V3 | PASS, max_abs=0.00105309 | 41.7561 / 41.7769 ms | 9.7829 / 9.8054 ms | 6,699,061 token/s | 4.268x |
| #13: B64/S1024/D128/H4/L4/FFN128 | V3.1 | PASS, max_abs=0.00105309 | 41.7584 / 41.7826 ms | 5.4255 / 5.4438 ms | 12,079,318 token/s | **7.697x** |

So với v3, v3.1 giảm optimized median `12.5%` ở shape #1 và `44.5%` ở shape #13. Shape #6 chỉ chạy với `accuracy-trials=1`, `warmup=0`, `repeats=1`, `rounds=1`; kết quả đó không được dùng làm performance report.

## 7. S4 — Shape-aware scheduler

**Trạng thái:** Idea.

### Giả thuyết

Không có một kernel tốt nhất cho mọi batch size, sequence length, hidden dimension, dtype và mask. Scheduler có thể chọn giữa baseline, SDPA, compile hoặc custom kernel theo shape.

### Thiết kế dự kiến

- Key: `(gpu_arch, dtype, B, S, D, H, causal, has_padding)`.
- Candidate implementation phải pass accuracy trước khi được đăng ký.
- Với key chưa biết, dùng safe fallback và có thể profile các candidate trong chế độ offline.
- Lưu lựa chọn bằng bảng tĩnh hoặc cache version hóa; không autotune trong đường inference chính thức nếu chi phí không được tính công bằng.

### Việc cần làm

- Chốt toàn bộ shape combinations của ban tổ chức.
- Tạo benchmark matrix runner.
- Xác định break-even point giữa packed SDPA, compile và custom kernel.

## 8. S5 — Low-precision optimized path

**Trạng thái:** V4 FP16 GPU official shape #1 benchmarked; V4 BF16 GPU accuracy
FAIL; full matrix pending.

### Giả thuyết

Sau whole-model compile, GEMM và attention chiếm phần lớn device time trên official
shape #1. RTX 5090 có throughput FP16/BF16 cao hơn FP32, nên chỉ hạ precision của
QKV, attention, output projection và hai FFN GEMM có thể giảm latency trong khi
LayerNorm, residual accumulation, GELU và final output vẫn ở FP32 để bảo vệ error
budget.

### Candidate

- `v4_FP16.py`: FP32 input/norm/residual/output; FP16 QKV, SDPA, output projection
  và FFN GEMM.
- `v4_BF16.py`: cùng graph nhưng dùng BF16 cho các compute-heavy operation.
- Low-precision weights/biases là non-persistent inference cache, refresh sau
  `load_state_dict()` và sau device/dtype transform.
- Training hoặc input dtype khác FP32 fallback toàn bộ về reference.
- GELU nhận kết quả FFN-in đã cast lại FP32 rồi mới cast xuống trước FFN-out.

### Correctness/benchmark gate

- Kiểm tra causal/non-causal × padding/no-padding, nhiều seed và strict
  `relative < 0.02 OR absolute < 0.002` trước performance.
- So riêng FP16 và BF16; candidate fail không được benchmark/promotion.
- Nếu cả hai pass, benchmark eager rồi `torch.compile(mode="default")` trên cùng
  official shape/config với V3.1.
- Full 14-shape matrix vẫn là điều kiện promote; local smoke không phải kết quả
  performance chính thức.

Không promote phương án nếu chỉ pass shape nhỏ hoặc một seed.

### Local diagnostic sau implementation

Môi trường local CPU, PyTorch `2.12.1`; performance không hợp lệ để báo cáo.
Official shape #2 được dùng đúng dimensions nhưng chỉ để accuracy smoke với ba
trial:

| Candidate | Accuracy | Max abs | Failed elements | Performance status |
|---|---|---:|---:|---|
| V4 FP16 internal | PASS | `0.00115311` | `0/49,152` | CPU timing ignored |
| V4 BF16 internal | **FAIL** | `0.00808978` | `1,569/49,152` | Correctly skipped by gate |

FP16 còn PASS local causal/non-causal × padding/no-padding trên ba seed ở
diagnostic shape `B2/S17/D32/H4/FFN48/L4` (`0/13,056` failed,
`max_abs=0.00141492`) và PASS `torch.compile(mode="default")` smoke. BF16 cùng
diagnostic matrix fail `532/13,056`, nên BF16 chưa phải candidate hợp lệ; file
được giữ để xác nhận lại backend CUDA và làm negative ablation, không được dùng
cho performance claim nếu GPU accuracy vẫn fail.

### GPU official shape #1

```bash
CUDA_VISIBLE_DEVICES=1 python -m tools.profile_models \
  --impl v4_FP16.py v4_BF16.py --shape-id 1 \
  --accuracy-trials 5 \
  --warmup 50 --repeats 200 --benchmark-rounds 5
```

Môi trường: RTX 5090 vật lý index `1`, PyTorch `2.13.0+cu130`, CUDA `13.0`,
FP32 public input/output, optimized eager, TF32 policy giống baseline.

| Candidate | Accuracy | Baseline median | Optimized median / p90 | Speedup |
|---|---|---:|---:|---:|
| V4 FP16 | PASS, 5 trials | 1.1012 ms | 0.6191 / 0.6636 ms | 1.779x |
| V4 BF16 | **FAIL** | — | — | — |

V4 FP16 giảm profiler self time của GEMM từ `0.1304` xuống `0.0835 ms` và
attention từ `0.1175` xuống `0.0530 ms` so với profile V3.1 trước đó. Tuy nhiên
`aten::copy_` tăng từ 5 calls / `0.0102 ms` lên 29 calls / `0.0694 ms`; tổng
GPU kernels là 100 mỗi forward. Vì vậy V4 FP16 vẫn chậm hơn V3.1 eager
`0.5118 ms` trong paired-like measurement và chưa được promote.

### S5.1 — FP16 GELU không round-trip qua FP32

**Trạng thái:** GPU official shape #1 eager/compiled benchmarked và năm official
shapes bổ sung PASS; full 14-shape matrix pending.

**Giả thuyết:** V4 FP16 giảm GEMM/attention time nhưng tạo 29 `aten::copy_` mỗi
forward trên official shape #1, so với 5 ở V3.1. Riêng việc cast FFN hidden
`FP16 → FP32` trước GELU rồi `FP32 → FP16` trước FFN-out tạo hai copy mỗi layer,
tức tám copy trên model bốn layer. Chạy GELU trực tiếp trên FP16 có thể bỏ các
round-trip này và giảm layout-copy/kernel-launch overhead.

**Phạm vi ablation:** Tạo candidate version mới, giữ nguyên FP32 LayerNorm,
residual accumulation và output; chỉ đổi dtype của GELU từ FP32 sang FP16.
Candidate phải pass strict correctness trước benchmark. So sánh V4 FP16 và
candidate mới trên cùng official shape #1 ở eager, sau đó thử
`torch.compile(mode="reduce-overhead")` nếu eager accuracy pass.

**Implementation:** `v4_1_FP16_GELU.py` dùng cùng cache/core với V4, nhưng đặt
`gelu_internal_dtype=True`. Control `v4_FP16.py` giữ nguyên GELU FP32. Alias
runner là `v4.1.fp16`.

**Standalone artifact:** `v4_1_clean.py` mirror đúng V4.1 model graph và cache
lifecycle nhưng không import benchmark harness, không CLI và không chứa
timing/accuracy code. File vẫn giữ parameter names/state dict cùng fallback
training/non-FP32 để có thể load strict weights chính thức. Đây là packaging
change, không phải optimization mới. Local PyTorch `2.12.1` xác nhận strict
state dict không có missing/unexpected keys; eager eval, training fallback và
BF16 fallback khớp V4.1 exact (`max_abs=0`). Local
`torch.compile(mode="reduce-overhead")` smoke PASS với `max_abs=4.76837e-07`.
Trên RTX 5090, standalone eager khớp V4.1 exact (`max_abs=0`) và compiled smoke
PASS strict comparator (`max_abs=2.38419e-07`). Các smoke shapes là diagnostic;
performance headline vẫn lấy từ benchmarkable `v4_1_FP16_GELU.py`.

**Local correctness (CPU, PyTorch 2.12.1):** causal/non-causal ×
padding/no-padding trên diagnostic shape `B2/S17/D32/H4/FFN48/L4`, ba trial mỗi
nhánh, đều PASS (`failed=0/13,056`, max absolute error lớn nhất `0.00132047`).
Official shape #2 PASS 5/5 (`max_abs=0.00115311`, `failed=0/81,920`). V4 control
và V4.1 cho cùng accuracy summary trên case này. `torch.compile(mode=
"reduce-overhead")` local causal + padding smoke cũng PASS 3/3
(`max_abs=0.00177431`, `failed=0/3,264`). CPU latency không được dùng làm
performance result.

**GPU eager ablation, official shape #1:** RTX 5090 vật lý index `1`, PyTorch
`2.13.0+cu130`, accuracy trials `5`, warmup `50`, repeats `200`, rounds `5`:

```bash
CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m tools.profile_models \
  --impl v3_1_CausalMask.py v4_FP16.py v4_1_FP16_GELU.py \
  --shape-id 1 --accuracy-trials 5 \
  --warmup 50 --repeats 200 --benchmark-rounds 5
```

| Candidate | Accuracy | Baseline median | Optimized median / p90 | Speedup |
|---|---|---:|---:|---:|
| V3.1 | PASS, max_abs `0.00105309` | 1.0863 ms | 0.5136 / 0.5280 ms | 2.115x |
| V4 FP16, GELU FP32 | PASS, max_abs `0.00159800` | 1.0293 ms | 0.5836 / 0.6433 ms | 1.764x |
| V4.1 FP16 GELU | PASS, max_abs `0.00159800` | 1.0235 ms | **0.5492 / 0.5880 ms** | 1.864x |

V4.1 giảm V4 eager median `5.9%`. Profiler xác nhận `aten::copy_` giảm từ 29
xuống 21 calls/fwd, layout-copy self time `0.0695 → 0.0499 ms`, ATen calls
`92 → 84` và GPU kernels `100 → 92`. GELU self time cũng giảm `0.0086 →
0.0065 ms`. V4.1 vẫn chậm hơn V3.1 eager `6.9%`, nên thay đổi này chỉ giải
quyết một phần conversion overhead.

**GPU compile ablation, cùng shape/config:**

```bash
CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m tools.profile_models \
  --impl v3_1_CausalMask.py v4_FP16.py v4_1_FP16_GELU.py \
  --shape-id 1 --accuracy-trials 5 \
  --warmup 50 --repeats 200 --benchmark-rounds 5 \
  --compile-user --compile-mode reduce-overhead
```

| Candidate compiled | Accuracy | Baseline median | Optimized median / p90 | Speedup |
|---|---|---:|---:|---:|
| V3.1 | PASS, max_abs `0.00105298` | 1.0289 ms | 0.3168 / 0.3189 ms | 3.247x |
| V4 FP16 | PASS, max_abs `0.00159800` | 1.0889 ms | **0.1858 / 0.1860 ms** | 5.861x |
| V4.1 FP16 GELU | PASS, max_abs `0.00159800` | 1.0828 ms | **0.1858 / 0.1860 ms** | 5.828x |

V4 và V4.1 compiled có cùng latency, 34 GPU kernels, 14 Triton GPU events, một
compiled region và một CUDA Graph launch mỗi forward. Inductor fuse dtype casts
quanh GELU vào cùng Triton region, nên manual GELU-FP16 ablation không tạo gain
thêm sau compile. Mixed precision compiled vẫn giảm candidate median `41.4%` so
với V3.1 compiled trên cùng run (`0.3168 → 0.1858 ms`).

**Selected official-shape matrix, V4.1 eager:** accuracy trials `5`, warmup `20`,
repeats `100`, rounds `3`; mọi row `failed=0`.

| Shape | Max abs | Baseline median | V4.1 median / p90 | Speedup |
|---:|---:|---:|---:|---:|
| #2 | 0.00108945 | 1.0311 ms | 0.5470 / 0.5591 ms | 1.885x |
| #7 | 0.00188218 | 1.0289 ms | 0.5545 / 0.6154 ms | 1.856x |
| #8 | 0.00160062 | 6.2593 ms | 3.2511 / 3.2536 ms | 1.925x |
| #12 | 0.00137764 | 1.0831 ms | 0.6111 / 0.6409 ms | 1.772x |
| #13 | 0.00149554 | 41.7541 ms | **2.7813 / 2.7996 ms** | **15.013x** |

Artifact GPU: `runs/profiles/profile_shape01_20260828T063013Z.json`,
`runs/profiles/profile_shape01_20260828T063051Z.json` và
`runs/benchmarks/matrix_v4_1_FP16_GELU_float32_20260828T063157Z.{json,csv}`
trên host `/home/chim/techjam-2026-track3`; các artifact đã được tải về cùng thư
mục tương ứng trong local workspace và bị `.gitignore` loại khỏi commit mặc định.

### S5.2 — Forced SDPA backend sweep

**Trạng thái:** GPU sweep complete cho official shapes #1–#13; shape #14 pending.

**Giả thuyết:** V4.1 compiled trace trên official shape #1 dùng
`fmha_cutlassF_f16`, nhưng automatic SDPA dispatch chưa chứng minh đây là backend
nhanh nhất trên RTX 5090 cho mọi official shape. Ép riêng Flash Attention,
memory-efficient attention và cuDNN attention trên cùng V4.1 graph có thể tìm
backend tốt hơn, đặc biệt ở attention-heavy shape #13.

**Phạm vi:** `v4_2_SDPA_Flash.py`, `v4_2_SDPA_Efficient.py` và
`v4_2_SDPA_CuDNN.py` chỉ thay backend context quanh SDPA. FP16 cache, QKV layout,
GELU, LayerNorm, residual, masking, correctness comparator và compile mode giữ
nguyên. `v4_1_FP16_GELU.py` là automatic-dispatch control. Backend không hỗ trợ
shape/mask phải được ghi là unsupported/error, không được fallback ngầm sang
backend khác.

**Backend compatibility:** Flash fail trước accuracy vì PyTorch `2.13.0+cu130`
báo `Flash Attention does not support non-null attn_mask`. Forced Efficient PASS
và dùng cùng `fmha_cutlassF_f16` như automatic V4.1. Forced cuDNN PASS các shape
có head dimension tối đa `128` đã thử nhưng không hỗ trợ official #8
(`head_dim=256`). Causal và non-causal với `padding_ratio=0.25` đều PASS 5/5 trial
trên diagnostic shape #1; max abs lần lượt `0.00159800` và `0.00111717`.

**Paired profiler ablation:** RTX 5090, PyTorch `2.13.0+cu130`, CUDA `13.0`,
FP32 public/FP16 internal, `torch.compile(mode="reduce-overhead")`:

| Shape | Backend | Accuracy | Optimized median / p90 | Device time | Speedup |
|---:|---|---|---:|---:|---:|
| #1 | Automatic/Efficient | PASS, max_abs `0.00159800` | 0.1858 / 0.1860 ms | 0.1819 ms | 5.590x |
| #1 | cuDNN | PASS, max_abs `0.00159800` | **0.1716 / 0.1736 ms** | 0.1685 ms | **5.940x** |
| #13 | Automatic | PASS, max_abs `0.00123096` | 2.3126 / 2.3267 ms | 2.3094 ms | 18.052x |
| #13 | Efficient | PASS, max_abs `0.00123096` | 2.3172 / 2.3332 ms | 2.3146 ms | 18.019x |
| #13 | cuDNN | PASS, max_abs `0.00121939` | **1.6841 / 1.6880 ms** | 1.6811 ms | **24.799x** |

Ở #13, cuDNN attention kernel giảm từ `1.2878` xuống `0.9590 ms`; candidate
end-to-end giảm `27.2%`. Ở #1, end-to-end giảm `7.7%`. Kernel count vẫn `34`,
compiled region `1` và CUDA Graph launch `1` mỗi forward, nên gain đến từ device
kernel chứ không phải giảm launch count.

**V4.2 dispatcher:** `v4_2_SDPA_Dispatch.py` ép cuDNN cho
#1/#2/#3/#4/#7/#9/#13, là các rows có win rõ; #5/#6/#8/#10/#11/#12 và shape
khác dùng automatic fallback. Dispatch dựa trên config khi khởi tạo, trước
compile. Command matrix:

```bash
CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m tools.matrix_runner \
  --impl v4.2.dispatch \
  --shape-ids 1,2,3,4,5,6,7,8,9,10,11,12,13 \
  --device cuda:0 --dtype float32 \
  --compile-user --compile-mode reduce-overhead
```

Accuracy trials `5`, warmup `20`, repeats `100`, rounds `3`; mọi row dưới đây
`failed=0`:

| ID | Max abs | Baseline median | V4.2 median / p90 | Speedup |
|---:|---:|---:|---:|---:|
| 1 | 0.00159800 | 1.0306 ms | 0.1715 / 0.1734 ms | 6.009x |
| 2 | 0.00108945 | 1.0499 ms | 0.0709 / 0.0801 ms | 14.812x |
| 3 | 0.00114536 | 1.0593 ms | 0.0730 / 0.0791 ms | 14.513x |
| 4 | 0.00137758 | 1.0362 ms | 0.1304 / 0.1310 ms | 7.944x |
| 5 | 0.00162899 | 1.0988 ms | 0.2719 / 0.2740 ms | 4.041x |
| 6 | 0.00184646 | 176.9139 ms | 26.2949 / 26.3644 ms | 6.728x |
| 7 | 0.00188218 | 1.0360 ms | 0.0977 / 0.0980 ms | 10.601x |
| 8 | 0.00173044 | 6.2819 ms | 2.7643 / 2.7745 ms | 2.273x |
| 9 | 0.00144446 | 0.9493 ms | 0.1675 / 0.1694 ms | 5.669x |
| 10 | 0.00140238 | 1.0306 ms | 0.1736 / 0.1756 ms | 5.936x |
| 11 | 0.00140238 | 1.5936 ms | 0.2646 / 0.2698 ms | 6.023x |
| 12 | 0.00137758 | 1.0533 ms | 0.1306 / 0.1309 ms | 8.066x |
| 13 | 0.00149554 | 41.7653 ms | **1.6840 / 1.6881 ms** | **24.801x** |

Geometric mean của 13 speedups là `7.58x`, so với khoảng `7.09x` của V4.1
matrix trước đó, tăng `6.9%`. Đây vẫn là partial matrix vì shape #14 chưa chạy.
Artifacts local/GPU:
`runs/benchmarks/matrix_v4_2_SDPA_Dispatch_float32_20260828T074434Z.{json,csv}`,
`runs/profiles/profile_shape01_20260828T073753Z.json` và
`runs/profiles/profile_shape13_20260828T073619Z.json`.

### S5.3 — Causal key-mask elision + Flash Attention

**Trạng thái:** Historical static-dispatch ablation benchmarked; superseded bởi S5.4.

**Giả thuyết:** Harness truyền `valid_token_mask` toàn `True` khi
`padding_ratio=0`, khiến V4.2 tạo non-null `attn_mask` và PyTorch Flash từ chối
kernel. Với causal self-attention và right padding, key mask còn dư cả khi có
padding: mọi valid query nằm trước padded keys nên causal constraint đã chặn các
key đó; invalid query outputs vẫn được zero cuối block.

**Implementation:** `v4_3_SDPA_Flash_NoMask.py` là forced-Flash ablation bỏ
riêng attention key mask. `v4_3_SDPA_CausalFlash_Dispatch.py` chọn Flash cho
#1/#4/#5/#7/#8/#10/#11/#13, cuDNN masked cho #2/#3/#9 và automatic masked cho
#6/#12/shape lạ. Shape key là Python constant trước compile nên hot path không
thêm tensor branch hoặc host sync. Optimization giả định mask hợp lệ có dạng
prefix `True` rồi suffix `False`; arbitrary masks có lỗ ở giữa không thuộc fast
path đã chứng minh.

**Correctness bổ sung:** causal shape #1 với `padding_ratio=0.25` PASS 5/5 trial
(`max_abs=0.001598`, `failed=0/5,242,880`); `padding_ratio=0.75` PASS 3/3 trial
(`max_abs=0.001598`, `failed=0/3,145,728`). Non-causal và shape lạ giữ masked
automatic fallback.

**Paired profiler:** RTX 5090, PyTorch `2.13.0+cu130`, FP32 public/FP16 internal,
`torch.compile(mode="reduce-overhead")`:

| Shape | V4.2 median | Flash no-mask median | Attention kernel V4.2 → Flash | Giảm end-to-end |
|---:|---:|---:|---:|---:|
| #1 | 0.1735 ms | **0.1490 ms** | 0.0443 → 0.0212 ms | 14.1% |
| #13 | 1.6843 ms | **1.1401 ms** | 0.9607 → 0.4232 ms | 32.3% |

Trace xác nhận event `pytorch_flash::flash_fwd_kernel`; Flash path còn 33 GPU
kernels so với 34 ở V4.2 và vẫn có một compiled region/CUDA Graph launch.

**Full dispatcher matrix #1–#13:** accuracy trials `3`, warmup `20`, repeats
`100`, rounds `3`; mọi row `failed=0`:

```bash
CUDA_VISIBLE_DEVICES=1 python -m tools.matrix_runner \
  --impl v4_3_SDPA_CausalFlash_Dispatch.py \
  --shape-ids 1,2,3,4,5,6,7,8,9,10,11,12,13 \
  --device cuda:0 --dtype float32 \
  --accuracy-trials 3 --warmup 20 --repeats 100 --benchmark-rounds 3 \
  --compile-user --compile-mode reduce-overhead
```

| ID | Max abs | Baseline median | V4.3 median | Speedup |
|---:|---:|---:|---:|---:|
| 1 | 0.00159800 | 1.0376 ms | 0.1489 ms | 6.969x |
| 2 | 0.000923514 | 1.0354 ms | 0.0719 ms | 14.393x |
| 3 | 0.00114536 | 1.0924 ms | 0.0771 ms | 14.173x |
| 4 | 0.00133067 | 1.0311 ms | 0.1222 ms | 8.435x |
| 5 | 0.00159800 | 1.0972 ms | 0.2185 ms | 5.022x |
| 6 | 0.00184646 | 176.9087 ms | 26.2940 ms | 6.728x |
| 7 | 0.00188218 | 1.0448 ms | 0.0791 ms | 13.203x |
| 8 | 0.00167191 | 6.2754 ms | 2.6423 ms | 2.375x |
| 9 | 0.00144446 | 0.9477 ms | 0.1674 ms | 5.660x |
| 10 | 0.00126258 | 1.0185 ms | 0.1510 ms | 6.746x |
| 11 | 0.00123510 | 1.5675 ms | 0.1940 ms | 8.079x |
| 12 | 0.00133711 | 1.0646 ms | 0.1304 ms | 8.166x |
| 13 | 0.00130266 | 41.7724 ms | **1.1412 ms** | **36.604x** |

Geomean speedup #1–#13 là `8.48x`, tăng `11.9%` so với V4.2 `7.58x`.
Artifacts:
`runs/benchmarks/matrix_v4_3_SDPA_CausalFlash_Dispatch_float32_20260828T081745Z.{json,csv}`,
`runs/profiles/profile_shape01_20260828T080522Z.json` và
`runs/profiles/profile_shape13_20260828T080628Z.json`. Shape #14 chưa chạy và
không bị loại khỏi correctness requirement.

### S5.4 — V4.3 Flash-first đơn giản hóa

**Trạng thái:** GPU benchmarked trên official shapes #1–#13; shape #14 pending.

**Giả thuyết:** Forced Flash đã PASS #1–#13 và geomean ghép từ sweep là khoảng
`8.49x`, gần như bằng static dispatcher `8.48x`. Bảng shape dài vì thế không tạo
lợi ích aggregate đáng kể, trong khi làm solution khó đọc và phụ thuộc RTX 5090
cụ thể hơn mức cần thiết.

**Phạm vi:** `v4_3_Flash.py` bỏ static shape registry. Mọi causal/right-padding
optimized path truyền `attn_mask=None` và ưu tiên backend theo thứ tự
Flash → cuDNN → Efficient → Math bằng `sdpa_kernel(..., set_priority=True)`.
Backend không đủ điều kiện được PyTorch bỏ qua; non-causal giữ key mask và
automatic dispatch. Training hoặc public input không phải FP32 vẫn dùng full
reference fallback của core V4.

**Standalone packaging hypothesis:** Tạo `v4_3_flash_clean.py` bằng cách mirror
toàn bộ config/model, parameter names, non-persistent FP16 cache lifecycle,
Flash-first causal path và safe fallback của V4.3, nhưng bỏ dependency vào
`torch_transformer_benchmark` cùng benchmark CLI. Packaging chỉ hợp lệ nếu
strict state-dict load, eager output, training/non-FP32 fallback và các nhánh
causal/non-causal có mask khớp implementation benchmarkable; thay đổi này không
tạo performance claim mới.

**Standalone packaging validation:** Local PyTorch `2.12.1` PASS syntax và
strict state-dict load, giữ mọi inference cache ở FP16 sau `load_state_dict()`
và `.to(dtype=torch.float32)`. Eager causal/non-causal × padding/no-padding,
training fallback và BF16 fallback đều khớp `v4_3_Flash.py` exact
(`rtol=0`, `atol=0`). Strict comparator còn PASS ba trial trên official shape
#2 causal/no-padding (`max_abs=0.00105053`) và diagnostic causal/non-causal có
right padding (`max_abs` lớn nhất `0.00102714`). Local
`torch.compile(mode="reduce-overhead")` smoke PASS (`max_abs=0.000219792` so
với eager). Đây là equivalence/correctness smoke cho packaging, không phải GPU
benchmark hoặc official-shape performance result.

**Validation:** Representative #1/#6/#8/#13 PASS trước full run. Direct matrix
#1–#13 dùng accuracy trials `3`, warmup `20`, repeats `100`, rounds `3`; mọi row
`failed=0`:

```bash
CUDA_VISIBLE_DEVICES=1 python -m tools.matrix_runner \
  --impl v4.3 --shape-ids 1,2,3,4,5,6,7,8,9,10,11,12,13 \
  --device cuda:0 --dtype float32 \
  --accuracy-trials 3 --warmup 20 --repeats 100 --benchmark-rounds 3 \
  --compile-user --compile-mode reduce-overhead
```

| ID | Max abs | Baseline median | V4.3 Flash median | Speedup |
|---:|---:|---:|---:|---:|
| 1 | 0.00159800 | 1.0358 ms | 0.1489 ms | 6.957x |
| 2 | 0.000923514 | 1.0307 ms | 0.0699 ms | 14.748x |
| 3 | 0.00114536 | 1.0796 ms | 0.0719 ms | 15.008x |
| 4 | 0.00133067 | 1.0361 ms | 0.1222 ms | 8.480x |
| 5 | 0.00159800 | 1.0990 ms | 0.2188 ms | 5.023x |
| 6 | 0.00184646 | 176.9036 ms | 26.9589 ms | 6.562x |
| 7 | 0.00188218 | 1.0501 ms | 0.0791 ms | 13.270x |
| 8 | 0.00167191 | 6.2962 ms | 2.6526 ms | 2.374x |
| 9 | 0.00144446 | 0.9452 ms | 0.1654 ms | 5.715x |
| 10 | 0.00126258 | 1.0296 ms | 0.1510 ms | 6.820x |
| 11 | 0.00123510 | 1.5672 ms | 0.1940 ms | 8.081x |
| 12 | 0.00133710 | 1.0393 ms | 0.1306 ms | 7.957x |
| 13 | 0.00130266 | 41.7726 ms | **1.1412 ms** | **36.604x** |

Geomean direct là `8.52x`, nhỉnh hơn static dispatcher `8.48x`. #6 chịu
regression khoảng `2.5%`, nhưng bảng shape được loại bỏ và aggregate không giảm.
Compiled profiler #13 báo `pytorch_flash::flash_fwd_kernel`, 33 GPU kernels, một
compiled region và một CUDA Graph launch; optimized median profiler `1.1453 ms`.

**Fallback diagnostics:** causal CPU (không có CUDA Flash) PASS 2/2 qua Math
fallback, `max_abs=0.000629663`; non-causal + 25% padding PASS 2/2 qua masked
automatic path, `max_abs=0.000522316`.

Artifacts:
`runs/benchmarks/matrix_v4_3_Flash_float32_20260828T083614Z.{json,csv}` và
`runs/profiles/profile_shape13_20260828T083514Z.json`.

### S5.5 — V5 FP8 GEMM với FP16 attention

**Trạng thái:** Rejected for promotion; kernel probes valid nhưng mọi model
candidate đều FAIL accuracy, vì vậy không benchmark performance.

**Giả thuyết:** RTX 5090 hỗ trợ FP8 Tensor Core GEMM. Sau V4.3, có thể giảm
thêm thời gian của packed QKV, attention-output và hai FFN projection bằng
`torch._scaled_mm` với E4M3, trong khi giữ LayerNorm/residual/output ở FP32 và
Flash SDPA ở FP16. Weight được quantize/cache một lần; activation dùng dynamic
scale trong từng forward. Bias được cộng sau GEMM vì target runtime không hỗ trợ
bias trực tiếp cho FP8 FP16-output GEMM này.

**Probe trên target:** RTX 5090 (SM 12.0), PyTorch `2.13.0+cu130` có
`torch.float8_e4m3fn`, `torch._scaled_mm` và FP8 per-tensor GEMM hoạt động.
SDPA không nhận FP8 Q/K/V (`mul_cuda` chưa implement cho E4M3), nên attention
không thể hạ trực tiếp xuống FP8. Row-wise FP8 scaling trên đúng runtime này trả
sai ngay sanity check all-ones (`32×128 @ 128×64` cho `3.5` thay vì `128`), nên
candidate không dùng row-wise path. `torchao` chưa được cài và không cần cho
probe built-in này.

**Gate:** Tạo version riêng `v5_FP8.py`; chạy strict comparator trên official
shape nhỏ trước. Nếu per-tensor FP8 fail accuracy thì ghi candidate là negative
ablation và không benchmark. Chỉ khi accuracy pass mới thử compile/profile và
mở rộng official matrix.

**Kết quả per-tensor:** `v5_FP8.py` chạy được trên official shape #2 nhưng FAIL
3/3 trial: `max_abs=0.110912`, `failed=25,429/49,152`. Performance bị accuracy
gate bỏ qua. Quantization per-tensor của cả năm Linear mỗi layer dùng quá nhiều
error budget và không phải candidate hợp lệ.

**V5.1-MXFP8 hypothesis:** Blackwell còn có native MXFP8 block scaling: mỗi block 32
phần tử dùng E8M0 power-of-two scale và scale tensor được swizzle theo layout
32×4×4. Phương án này giữ nhiều dynamic range hơn per-tensor FP8 và tránh
row-wise path đang sai. Thử `v5_1_MXFP8.py` bằng public
`torch.nn.functional.scaled_mm`; vẫn giữ SDPA FP16. Candidate phải pass all-ones
kernel sanity check rồi strict model accuracy trước benchmark.

**Kết quả MXFP8:** Sanity GEMM `ones[32,128] @ ones[128,384]` trả đúng `128` ở
mọi phần tử, xác nhận quantization, E8M0 scale và swizzle layout đúng. Tuy nhiên
full model official shape #2 vẫn FAIL 3/3 trial (`max_abs=0.144848`,
`failed=25,800/49,152`). Các ablation hẹp đều giữ V4.3 FP16 cho phần còn lại:

| MXFP8 scope qua 4 layers | Trials | Max abs | Failed elements | Gate |
|---|---:|---:|---:|---|
| Packed QKV only | 3 | `0.066844` | `7,532/49,152` | FAIL |
| FFN-in + FFN-out | 3 | `0.111659` | `24,477/49,152` | FAIL |
| Attention output only | 1 | `0.0698606` | `2,951/16,384` | FAIL |
| FFN input only | 1 | `0.0627175` | `6,380/16,384` | FAIL |
| FFN output only | 1 | `0.0607120` | `6,273/16,384` | FAIL |

Files `v5_2_MXFP8_*.py` giữ các single-scope ablation để tái lập kết luận. Vì
không biến thể nào qua strict correctness gate, repository không chạy hoặc ghi
speedup của V5/V5.1-MXFP8/V5.2. Muốn quay lại FP8 cần calibration/QAT, scale recipe
khác đã được target runtime xác nhận đúng, hoặc dispatch chỉ cho một shape/layer
đã pass đầy đủ; không được promote chỉ dựa trên peak FP8 throughput.

### S5.6 — V5.1 full FP16 accumulation

**Trạng thái:** Rejected for promotion; max-autotune fail strict accuracy ở
official shape #10, còn reduce-overhead fail mạnh ở official shape #8.

**Giả thuyết:** V4.3 đã hạ các Linear nặng xuống FP16 nhưng PyTorch mặc định vẫn
accumulate FP16 GEMM bằng FP32. Trên RTX 5090, bật process-global
`torch.backends.cuda.matmul.allow_fp16_accumulation = True` có thể giảm thời
gian QKV, attention-output và hai FFN GEMM, đổi lại một phần error budget.

Candidate `v5_1_FP16Accum.py` kế thừa nguyên graph, weight cache, Flash-first
SDPA và FP32 public contract của V4.3; chỉ policy accumulation của CUDA GEMM
thay đổi. Matrix/profile runner chạy mỗi implementation trong subprocess riêng,
nên flag global không rò sang candidate khác. Baseline public FP32 GEMM không bị
flag FP16 này tác động.

Tên lịch sử của file MXFP8 cũ được ghi rõ là `V5.1-MXFP8`; alias runnable
`v5.1` nay trỏ tới `v5_1_FP16Accum.py`. Không benchmark nếu strict accuracy fail.
Đo cả `reduce-overhead` và `max-autotune` vì Inductor có thể chọn kernel khác
nhau theo compile mode.

**Kết quả max-autotune #1–#13:** Với accuracy trials `5`, warmup `20`, repeats
`100`, rounds `3`, 12/13 shapes PASS. Shape #10 FAIL ở trial 5 với đúng
`1/5,242,880` phần tử fail: baseline `-0.068714015`, optimized `-0.066627771`,
`abs=0.00208624` và relative khoảng `3.0%`; benchmark row này bị gate bỏ qua.
Vì matrix không qua correctness nên không tính geomean V5.1 hợp lệ.

Paired profile trên shape #8, max-autotune, warmup `50`, repeats `200`, rounds
`5` cho V4.3 `2.5258 ms` và V5.1 `2.5336 ms`; cả hai có 32 GPU events, 28
Triton GPU events và cùng kernel mix. Chênh `-0.3%` là noise/regression, không
có bằng chứng speedup từ FP16 accumulation sau autotune.

**Kết quả reduce-overhead #8:** FAIL cả 5 trial, `max_abs=0.00657034` và
`40,029/41,943,040` phần tử fail. Performance bị skip đúng theo gate. Điều này
cho thấy CUTLASS/cuBLAS path dùng error budget lớn hơn đáng kể so với graph
max-autotune.

Artifacts trên server:
`runs/benchmarks/matrix_v5_1_FP16Accum_float32_20260828T125701Z.json`,
`runs/benchmarks/matrix_v5_1_FP16Accum_float32_20260828T130403Z.json` và
`runs/profiles/profile_shape08_20260828T130209Z.json`.

**Kết luận:** Giữ file và alias để tái lập negative ablation, nhưng không đưa
V5.1 vào scheduler/bảng speedup. V4.3 + max-autotune vẫn là đường chạy tốt nhất.

### S5.7 — V6 approximate GELU tanh

**Trạng thái:** Accuracy validated #1–#13; không promote vì clean paired
max-autotune không cho gain nhất quán và official #8 còn regression `0.32%`.

**Giả thuyết:** V4.3 gọi `F.gelu(..., approximate="none")` trên FP16 hidden.
Thay bằng `approximate="tanh"` có thể giảm chi phí activation/epilogue trong
Inductor max-autotune graph mà không thay GEMM, SDPA, LayerNorm, residual,
masking, cache hoặc public FP32 contract.

Candidate `v6_ApproxGELU.py` phải kế thừa V4.3 và chỉ override `_mixed_ffn()` để
đổi đúng GELU approximation. Gate đầu tiên là official #7 và #10 với năm trial
vì đây là các shape có error margin đáng lo nhất trong lịch sử V4/V5.1. Chỉ nếu
cả hai PASS mới chạy #1–#13 và paired profile V4.3/V6 bằng max-autotune. Mọi row
FAIL phải dừng trước benchmark; shape #14 vẫn pending độc lập.

**Correctness:** Local CPU official #2 PASS 5/5 (`max_abs=0.0012275`) và
diagnostic causal/non-causal × padding/no-padding PASS 3/3 mỗi nhánh. Trên RTX
5090, max-autotune official #1–#13 đều PASS năm trial, `failed=0`; max absolute
error lớn nhất là `0.00214118` ở #7. Giá trị này vượt absolute threshold nhưng
các phần tử tương ứng vẫn đạt relative `<2%`, vì vậy strict OR comparator PASS.

**Performance caveat:** Trong full matrix có một ResNet training process khác
chiếm khoảng `53–80%` GPU vật lý #1. Baseline và các shape lớn cùng chậm bất
thường (#6/#8 gần gấp đôi artifact V4.3 idle trước đó), nên không dùng latency,
speedup hoặc geomean của matrix này làm performance claim.

**Clean paired diagnostics sau khi training workload kết thúc:**

- Official #2, thứ tự V4.3 → V6: end-to-end median `0.0780 → 0.0749 ms` có vẻ
  nghiêng về V6, nhưng profiler device time lại `0.0554 → 0.0560 ms`. V6 có 33
  kernels/35 GPU events so với 32/34 của V4.3 vì tanh làm một GELU/epilogue
  không còn fuse vào template lân cận. Hai tín hiệu trái nhau nên không được
  công nhận là gain.
- Official #8, thứ tự V4.3 → V6: `2.5502 → 2.5584 ms`; V6 chậm hơn `0.32%`.
  Cả hai có 32 kernels, 28 Triton events và cùng CUDA Graph structure.

Các reverse-order retry sau đó bị một evaluation workload mới làm nhiễu (P90
và raw GPU time bất thường) nên bị loại. Những lượt trước đó chạy dưới training
contention cũng chỉ dao động quanh zero khi đảo thứ tự. Vì clean #8 regression
và clean #2 không tự nhất quán giữa host median/device time, không có evidence
rằng tanh GELU cải thiện end-to-end max-autotune. Candidate được giữ để tái lập
accuracy/fusion ablation nhưng không thay V4.3.

Artifacts trên server:
`runs/benchmarks/matrix_v6_ApproxGELU_float32_20260828T135007Z.json`,
`runs/profiles/profile_shape02_20260828T140504Z.json` và
`runs/profiles/profile_shape08_20260828T140546Z.json`. Reverse-order artifacts
`profile_shape08_20260828T140659Z.json` và `profile_shape02_20260828T141001Z.json`
bị loại khỏi claim vì evaluation contention.

### S5.8 — V7 residual + LayerNorm pipeline

**Trạng thái:** Rejected for promotion; V7a hạ xuống cùng compiled graph với
V4.3, không có gap để biện minh V7b custom Triton.

**Giả thuyết:** V4.3 trả attention/FFN projection về FP32 trước residual rồi
gọi LayerNorm/cast ở phép toán kế tiếp. Pipeline mỗi boundary thành
`(residual_fp32, normalized_fp16) = add_mask_layernorm(...)` có thể giúp
Inductor giữ branch FP16 tới điểm cộng, fuse cast + residual + optional mask +
LayerNorm và tránh materialize intermediate không cần thiết.

**Evidence trước thay đổi:** Compiled profiler đã có device events tên
`triton_per_fused__to_copy_add_native_layer_norm...` và biến thể chứa
`masked_fill`, nghĩa là Inductor đã fuse ít nhất một phần graph. Vì vậy V7a
trước hết chỉ pipeline pure PyTorch, không claim gain từ tên function. Candidate
chỉ đi tiếp nếu paired profile cho kernel/device-time khác V4.3.

**Phạm vi V7a:** Giữ nguyên weights/cache, exact FP16 GELU, Flash-first SDPA,
FP32 residual/LayerNorm/public output, mask semantics và fallback. Attention và
FFN branch giữ FP16; helper boundary cast branch lên FP32, cộng residual, áp
mask đúng vị trí, chạy LayerNorm FP32 rồi trả thêm activation FP16 cho GEMM kế
tiếp. Boundary cuối fuse về mặt graph với final LayerNorm; invalid output vẫn
được zero sau norm như reference.

**Gate:** Chạy local causal/non-causal × padding/no-padding và official #2;
sau đó GPU strict #7/#10 trước performance. Nếu PASS, paired max-autotune V4.3
với V7a trên #2/#8/#12. Chỉ triển khai V7b custom Triton nếu raw events cho thấy
V7a còn kernel/memory round trip đáng kể và có khả năng tạo gain lớn hơn noise.
Shape #14 vẫn pending độc lập.

**Correctness:** Local official #2 PASS 5/5 (`max_abs=0.00115311`); diagnostic
causal/non-causal × padding/no-padding PASS 3/3 mỗi nhánh, và V7a khớp V4.3
eager bit-for-bit trên cả bốn nhánh lẫn `valid_token_mask=None`. Training và
BF16 fallback cũng khớp exact, state-dict keys giữ nguyên. GPU max-autotune
#7/#10 PASS 5/5 với `max_abs` lần lượt `0.00188218` và `0.00140238`, bằng error
profile V4.3.

**Paired max-autotune trên GPU idle:**

- #2: V4.3 `0.0689 ms`, V7a `0.0699 ms`; raw GPU time cùng `0.0546 ms`.
  Cả hai có 32 kernels, 28 Triton events và cùng top-event names.
- #8: V4.3 `2.5155 ms`, V7a `2.5235 ms`; raw GPU time
  `2.4659 → 2.4713 ms`. V7a chậm hơn `0.32%`; cùng 32 kernels/28 Triton events.
- #12, V4.3 → V7a: `0.0813 → 0.0792 ms`, nhưng raw GPU time chỉ
  `0.0774 → 0.0771 ms`. Khi đảo thứ tự V7a → V4.3, median là
  `0.0812 → 0.0813 ms` và raw GPU time `0.0778 → 0.0775 ms`; effect đổi dấu
  theo metric và nằm trong noise. Cả hai có 33 kernels/29 Triton events.

Tên kernel trùng nhau, gồm
`triton_tem_fused__to_copy_addmm_native_layer_norm_t_view_1` và các kernel
`...add...masked_fill...native_layer_norm...`. Điều này xác nhận max-autotune đã
fuse cast/residual/mask/LayerNorm, thậm chí ghép LayerNorm với GEMM template ở
boundary phù hợp. Pure-PyTorch pipeline không đổi codegen. Viết một standalone
Triton add+LayerNorm lúc này có nguy cơ phá fusion mạnh hơn và không còn
profiler evidence để tiếp tục, nên V7b không được triển khai.

Artifacts trên server:
`runs/benchmarks/matrix_v7_ResidualLayerNorm_float32_20260828T142205Z.json`,
`runs/profiles/profile_shape02_20260828T142245Z.json`,
`runs/profiles/profile_shape08_20260828T142325Z.json`,
`runs/profiles/profile_shape12_20260828T142443Z.json` và
`runs/profiles/profile_shape12_20260828T142543Z.json`.

### S5.9 — V8 fused FFN-in GEMM + exact GELU epilogue

**Trạng thái:** Accepted có điều kiện theo shape; V8a được dispatch cho official
#6, mọi shape đã đo khác fallback V4.3. Shape #14 pending.

**Giả thuyết:** Max-autotune V4.3 vẫn có separate exact-GELU kernels trên nhiều
shape. Ở official #8, hai nhóm GEMM template chiếm khoảng `1.92/2.47 ms` raw
GPU time, còn Flash Attention khoảng `0.12 ms`. Fuse riêng FFN-in GEMM, bias và
exact GELU có thể bỏ một launch mỗi layer và loại round trip qua pre-GELU hidden
tensor mà không thay FFN-out hoặc model topology.

**Phạm vi V8a:** Custom Triton nhận normalized FP16 activation cùng cached
FFN-in FP16 weight/bias, accumulate dot product FP32, cộng bias, cast về FP16 tại
đúng rounding boundary của `F.linear`, tính exact/erf GELU từ giá trị đã round
rồi store FP16 cho FFN-out. FP32 LayerNorm/residual/public output, Flash-first
SDPA, exact GELU semantics, cache/state dict và training/non-FP32 fallback giữ
nguyên V4.3. Candidate dùng `torch.library.custom_op` để whole-model compile có
thể capture kernel; local không có Triton dùng exact PyTorch fallback.

**Gate:** Trước hết so custom kernel với compiled `F.linear → exact GELU` trên
representative FFN shapes #2/#6/#7/#8/#12. Chỉ nếu microkernel có accuracy hợp
lệ và ít nhất một measured win mới chạy model strict #7/#10, rồi paired
max-autotune V4.3/V8a. Candidate không được dispatch vào shape thua; full matrix
chỉ chạy sau khi targeted whole-model performance pass. Shape #14 vẫn pending.

**Microkernel:** So với compiled PyTorch exact path, custom kernel cho #2
`0.036803 → 0.036822 ms` (`0.999x`), #7 `0.042521 → 0.043244 ms`
(`0.983x`), #8 `0.243071 → 0.097361 ms` (`2.497x`) và #12
`0.041625 → 0.043057 ms` (`0.967x`). Sample #6 (`M=1,280,000`, `K=N=128`)
PASS comparator với `max_abs=0.00024414` và đạt `0.885853 → 0.869318 ms`
(`1.019x`). Microbenchmark #8 không chuyển thành whole-model win tương ứng,
nên không được dùng một mình để quyết định dispatch.

**Correctness:** Custom-math targeted #7/#10 PASS 5/5 với max abs lần lượt
`0.00188218` và `0.00140238`. Sau khi thêm static dispatcher, full official
#1–#13 max-autotune PASS 5/5 mỗi shape, `failed=0`; max abs lớn nhất
`0.00188218`. Artifact:
`runs/benchmarks/matrix_v8_FusedFFNGELU_float32_20260828T151419Z.json`.

**Paired whole-model:** Ở #8, forward V4.3/V8a dao động `2.5145/2.5154 ms`,
`2.5164/2.5145 ms`, rồi reverse-order `2.5341/2.5183 ms`; raw GPU time chỉ
dao động khoảng `±0.25%`. V8a giảm 32 xuống 29 kernels nhưng GEMM epilogue nặng
hơn bù lại launch/intermediate đã bỏ, nên #8 fallback V4.3.

Official #6 cho win lặp lại và không đổi dấu. Lượt đầu V4.3/V8a là
`26.6517/25.3846 ms` (`-4.75%` latency); reverse order là
`26.6879/25.3693 ms` (`-4.94%`). Sau khi thêm K=128 autotune configs và dispatch,
paired accuracy-5 cho `26.6799 → 25.4092 ms` (`-4.76%`), raw GPU time
`26.6680 → 25.3948 ms`, GPU kernels `32 → 29`, Triton events `28 → 21`.
V4.3 profiler có ba separate GELU launches/forward; V8 thay cả bốn
FFN-in/GELU operations bằng bốn custom-kernel calls và giảm tổng cộng ba kernel.

**Dispatch cuối:** bật V8a khi `B*S >= 1_000_000`, `D=128`, `FFN=128`; toàn bộ
config khác gọi lại V4.3 `_mixed_ffn`. Full #1–#13 matrix cho geomean speedup
`9.775x` so với eager baseline và #6 đạt `25.3436 ms`; đây là cross-shape
validation, còn performance claim chính dựa trên paired #6 ở GPU idle. Paired
artifact mới nhất:
`runs/profiles/profile_shape06_20260828T151245Z.json`.

### S5.10 — V8.1 force fused FFN/GELU trên mọi shape

**Trạng thái:** Completed; aggregate win nhỏ nhưng không promote thành
unconditional replacement vì #2/#12 order-sensitive và #11 raw GPU regression.

**Giả thuyết:** Static dispatcher V8 hiện chỉ bật custom kernel cho #6 dựa trên
paired evidence trước đó. Ép cùng kernel trên #1–#13 sẽ kiểm tra liệu autotune
configs mới có tạo win ở shape khác hay không, đồng thời lượng hóa regression
mà dispatcher đang tránh.

**Phạm vi:** `v8_1_FusedFFNGELUAll.py` kế thừa nguyên V8 nhưng đặt
`_use_fused_ffn_gelu=True` sau construction. Precision boundary, exact GELU,
cache/state dict, training/non-FP32 fallback và attention đều không đổi. V8
stable không bị sửa.

**Gate và metric:** Chạy strict accuracy 5 trial trên official #1–#13 trước.
Chỉ row PASS mới được benchmark. So paired V4.3/V8.1 trong cùng process bằng
`max-autotune`, cùng warmup/repeats/rounds; ghi median, raw GPU time và kernel
count. Kiểm tra GPU vật lý #1 idle trước run; rerun reverse order nếu effect gần
noise hoặc bất thường.

**Correctness:** GPU vật lý #1 idle, PyTorch `2.13.0+cu130`, FP32 public/FP16
internal. V8.1 PASS strict accuracy 5/5 trên official #1–#13; max abs lớn nhất
`0.00188218`, failed `0`. Correctness-gate artifact:
`runs/benchmarks/matrix_v8_1_FusedFFNGELUAll_float32_20260828T162406Z.json`.
Latency trong artifact này bị loại vì chỉ dùng warmup/repeats/rounds `1/1/1`.

**Paired max-autotune:** Mỗi direction dùng accuracy `5`, warmup `20`, repeats
`100`, rounds `3`; #6 dùng `10/20/3` do batch 10000. Giá trị dưới là phần trăm
V8.1 latency so với V4.3; số âm là nhanh hơn:

GPU process list trống trước cả hai paired sequences và trước #6. Một ResNet
training process chỉ bắt đầu lúc `23:41:38`, sau artifact cuối lúc `23:40:14`,
nên không overlap các measurement dưới đây.

| Shape | V4.3 → V8.1 | V8.1 → V4.3 | Raw GPU assessment |
|---:|---:|---:|---|
| 1 | -2.00% | -1.57% | -1.72% / -1.89% |
| 2 | +0.00% | +2.93% | -4.90% / -5.13%; host regression/order-sensitive |
| 3 | -1.42% | -1.47% | -6.84% / -6.53% |
| 4 | -2.73% | -2.69% | -2.33% / -2.90% |
| 5 | -1.52% | -1.21% | -1.31% / -1.59% |
| 6 | -4.78% | -4.94% từ prior reverse-order | khoảng -4.8% |
| 7 | +0.05% | -1.49% | khoảng -5.0%; host quantization/noise |
| 8 | -0.40% | -0.56% | -0.61% / -1.07% |
| 9 | -0.06% | -1.42% | -1.17% / -1.68% |
| 10 | -1.49% | -1.51% | -1.92% / -2.08% |
| 11 | -0.09% | -0.16% | **+0.41% / +0.52%** regression |
| 12 | -2.52% | **+2.47%** | -2.37% / -0.24%; effect đổi dấu |
| 13 | -0.66% | -0.92% | -0.31% / -0.62% |

Geomean latency ratio qua 13 shapes giảm `1.365%` ở direction đầu và `0.969%`
ở reverse direction. V8.1 luôn giảm khoảng 3–4 GPU kernels, nhưng #2/#12/#11
cho thấy kernel count không đủ để quyết định. #6 exact-wrapper rerun đạt
`26.6610 → 25.3857 ms`, raw GPU `26.6457 → 25.3668 ms`, kernels `32 → 29`;
artifact `runs/profiles/profile_shape06_20260828T163922Z.json`.

**Kết luận:** Force-all có aggregate gain khoảng `1%`, nhưng không thắng ổn
định từng shape. Giữ V8.1 làm ablation; không sửa V8 stable trong experiment
này. Hướng hợp lý tiếp theo là mở rộng dispatcher riêng cho các shape giữ dấu
win (#1/#3/#4/#5/#10, cùng #6 hiện tại), còn #2/#11/#12 giữ V4.3; #7/#8/#9/#13
cần measurement dài hơn nếu muốn claim gain dưới `1.5%`.

### S5.11 — V9 fully fused persistent MLP

**Trạng thái:** Completed ablation; correctness PASS nhưng không promote vì
whole-model không thắng V8/V4.3 ổn định.

**Giả thuyết:** V8 chỉ fuse FFN-in GEMM với GELU; FFN-out vẫn là GEMM riêng và
hidden FP16 vẫn được materialize. Với D/FFN nhỏ, một Triton program có thể giữ
hidden tiles on-chip, lập tức dot với FFN-out weight và chỉ store final FP16
projection. Việc bỏ hidden round trip cùng FFN-out launch có thể thắng dù tiled
kernel không đạt throughput của hai GEMM templates độc lập.

**Phạm vi V9a:** Custom op nhận normalized FP16 activation, hai cached FP16
weights/biases; FFN-in và FFN-out dot đều accumulate FP32. FFN-in+bias được
round FP16 trước exact erf-GELU, GELU được round FP16 trước FFN-out, và final
FFN-out+bias được round FP16 trước trả FP32, khớp ba precision boundaries V4.3.
Kernel tile output theo token/output dimension và loop FFN tiles mà không store
hidden. Support envelope đầu tiên là `D <= 128`, `FFN <= 128`, multiples of 16;
D=1024 và config lạ fallback V4.3.

**Gate:** So custom kernel với compiled `Linear → exact GELU → Linear` trên
M đại diện official #2/#12/#1/#5/#13 và D=FFN=32 của #7. Comparator dùng strict
contest rule cùng max-abs diagnostic. Chỉ nếu ít nhất một small-shape
microkernel win và model strict accuracy PASS mới chạy paired whole-model.
Không benchmark row accuracy fail; không promote chỉ vì kernel/launch count ít.

**Clean isolated microkernel sweep:** RTX 5090, GPU compute-process list trống,
FP16 input/weights, compiled PyTorch max-autotune reference, 30 warmup và
150–300 CUDA-event repeats:

| Official mapping | M / D / FFN | Max abs / failed | Reference | V9 kernel | Speedup |
|---|---|---|---:|---:|---:|
| #2 | 128 / 128 / 128 | 0 / 0 | 0.03968 ms | 0.03189 ms | 1.244x |
| #12/#4 | 2048 / 128 / 128 | 0 / 0 | 0.04639 ms | 0.03692 ms | 1.256x |
| #1/#9/#10/#11 | 8192 / 128 / 128 | 0 / 0 | 0.04272 ms | 0.03123 ms | 1.368x |
| #5 | 16384 / 128 / 128 | 0 / 0 | 0.04239 ms | 0.03190 ms | 1.329x |
| #13 | 65536 / 128 / 128 | 0 / 0 | 0.05320 ms | 0.04511 ms | 1.179x |
| #7 | 8192 / 32 / 32 | 0.00097656 / 0 | 0.04617 ms | 0.03194 ms | 1.445x |

Shape #6 isolated (`M=1,280,000`, sample 524,288 outputs) PASS với max abs
`0.00097656`; reference `1.29767 ms`, V9 `0.81689 ms` (`1.589x`). Hai lượt
microkernel trước đó bị concurrent evaluation làm nhiễu và bị loại hoàn toàn.

**Whole-model correctness:** V9 max-autotune PASS strict accuracy 5/5 trên
official #1–#13, max abs lớn nhất `0.00188218`, failed `0`. #8 D=1024 dùng
V4.3 fallback. Accuracy artifact
`runs/benchmarks/matrix_v9_PersistentMLP_float32_20260828T171104Z.json` dùng
warmup/repeats/rounds `1/1/1`; latency trong artifact này không hợp lệ để claim.

**Paired whole-model results:** GPU idle trước và sau sequences, accuracy `5`,
warmup/repeats/rounds `20/100/3` trừ reverse #7 `50/200/5` và #6 `10/20/3`:

| Shape | V4.3 | V9 | Host latency Δ | Raw GPU Δ | Kernels |
|---:|---:|---:|---:|---:|---:|
| 2 | 0.0689 ms | 0.0689 ms | 0.0% | +1.1% | 32 → 25 |
| 12 | 0.0812 ms | 0.0812 ms | 0.0% | -0.6% | 33 → 25 |
| 7 | 0.0709 ms | 0.0699 ms | -1.4% | -12.3% | 32 → 25 |
| 1 | 0.1325 ms | 0.1387 ms | **+4.7%** | +5.2% | 33 → 25 |
| 5 | 0.2084 ms | 0.2126 ms | **+2.0%** | +1.5% | 32 → 25 |
| 13 | 1.0891 ms | 1.1463 ms | **+5.3%** | +4.9% | 32 → 25 |

Reverse-order #7 đổi dấu theo primary latency: V9 `0.0709 ms`, V4.3
`0.0692 ms`, dù raw GPU vẫn `0.0583` vs `0.0665 ms`. Vì score-facing CUDA-event
median không giữ dấu, #7 không được claim là end-to-end win.

Shape #6 triple paired: V4.3 `26.6497 ms`, V8 `25.4071 ms`, V9
`25.4481 ms`; raw GPU lần lượt `26.6522/25.3974/25.4512 ms`, kernels
`32/29/25`. Full fusion nhanh hơn V4.3 `4.51%` nhưng chậm hơn V8 `0.16%`, nên
không thay current #6 dispatcher. Artifact:
`runs/profiles/profile_shape06_20260828T171913Z.json`.

**Kết luận:** V9 chứng minh full MLP fusion có thể nhanh hơn `1.18–1.59x` khi
đo isolated, nhưng whole graph đã khai thác GEMM/codegen fusion tốt hơn. D=128
regress khi workload đủ lớn; D=32 giảm raw events nhưng latency score nằm ở
timing floor và đổi dấu. Giữ V9 làm ablation, không dispatch trong best path.

### S5.12 — V10 persistent FFN-in + exact GELU

**Trạng thái:** Implementing; chưa có performance result.

**Giả thuyết:** V8 thắng ổn định trên official #6 khi fuse riêng FFN-in/GELU,
trong khi V9 chậm hơn V8 vì custom FFN-out không bằng CUTLASS trong whole graph.
Ở #6, `M=B*S=1,280,000` nhưng `D=FFN=128`, nên hàng nghìn independent M-tiles
dùng lại cùng FFN-in weight. Giới hạn grid quanh số SM và để mỗi CTA loop qua
nhiều M-tiles có thể giảm scheduling overhead và tái sử dụng weight tile tốt
hơn V8, trong khi vẫn giữ FFN-out bằng `F.linear` hiện tại.

**Phạm vi V10a:** Chỉ thay `FFN-in Linear → exact GELU` trên inference CUDA
FP16-internal path với `D=FFN=128` và ít nhất một triệu token rows. Dot vẫn
accumulate FP32; output Linear được round FP16 trước exact erf-GELU và GELU
store FP16, đúng precision boundaries V8/V4.3. Attention, FFN-out CUTLASS,
LayerNorm/residual FP32, cache/state dict và fallback giữ nguyên V8. Config khác
fallback V8/V4.3.

**Gate:** So isolated V8 và V10 kernel trên đúng mapping #6 trước, GPU idle,
cùng input/weights/warmup/repeats. Sau strict model accuracy #6, chạy paired
whole-model V8/V10 và reverse order bằng max-autotune. Chỉ promote khi primary
latency và raw GPU time cùng giữ dấu, gain vượt variance; kernel count hoặc
isolated win không đủ.

### S5.13 — V11 exact GELU trực tiếp từ FP32 accumulator

**Trạng thái:** Promoted main theo D-027; GPU #1–#13 strict PASS;
paired #6/#7/#10 completed; shape #14 pending.

**Giả thuyết:** V8/V10 đã accumulate FFN-in GEMM ở FP32 nhưng cố tình round
`Linear+bias` xuống FP16 rồi mới tính exact GELU. Baseline tính Linear/GELU ở
FP32, nên bỏ riêng lần round trước GELU có thể giảm mixed-precision error mà
không thêm launch, tensor trung gian hoặc memory traffic: kernel vẫn chỉ store
GELU output FP16 cho FFN-out.

**Phạm vi V11a:** Giữ nguyên activation/weight/bias FP16, FP32 dot
accumulation, exact erf-GELU, FP16 GELU output, FFN-out, attention,
LayerNorm/residual, cache/state dict và fallback của V8.1. Thay đổi duy nhất
trong custom Triton epilogue là tính GELU trực tiếp từ `accumulator + bias`
thay vì `FP32 → FP16 → FP32` trước GELU. Ép custom path trên mọi mixed-precision
shape để so accuracy và kernel latency trực tiếp với V8.1; dispatcher cuối chỉ
được chọn sau paired whole-model evidence.

**Gate:** Chạy syntax/local fallback trước, rồi strict GPU accuracy ưu tiên
official #7/#10/#6 và matrix #1–#13. Chỉ benchmark row đã PASS. So paired V8.1
với V11 bằng max-autotune trên cùng GPU idle, trước hết #7 và #6; yêu cầu host
latency/raw GPU time không regress vượt noise. Shape #14 vẫn là gate riêng.

**Local diagnostic:** PyTorch `2.12.1`, CPU. `py_compile`, runner alias/list,
state-dict equality, causal/non-causal × padding/no-padding và training fallback
đều PASS; public FP16/BF16 fallback khớp baseline exact. `torch.compile` capture
trên diagnostic B2/S16/D32/H4/FFN32/L2 causal PASS 2/2. CPU latency không được
dùng làm performance claim vì candidate đích là custom Triton CUDA.

Paired direct comparison dùng cùng baseline/weights/input cho V8.1 và V11 trên
10 seed official #7/#10; mọi row strict PASS:

| Shape | Metric | V8.1 | V11 | Nhận xét |
|---:|---|---:|---:|---|
| #7 | Mean absolute error | 0.00019114 | 0.00017636 | V11 tốt hơn 10/10 seed |
| #7 | Worst normalized OR risk | 0.802223 | 0.815028 | V11 hơi xấu hơn; chỉ tốt 4/10 seed |
| #10 | Mean absolute error | 0.00017845 | 0.00016247 | V11 tốt hơn 10/10 seed |
| #10 | Worst normalized OR risk | 0.718771 | 0.654384 | V11 tốt hơn 8/10 seed |

Normalized risk mỗi phần tử là minimum giữa `abs_error/0.002` và
`abs_error/(0.02*abs(reference))`; candidate strict PASS khi maximum risk `<1`.
Kết quả local ủng hộ giảm average error nhưng chưa chứng minh Pareto improvement
ở worst comparator margin #7.

**GPU environment:** RTX 5090, driver `595.58.03`, PyTorch `2.13.0+cu130`, CUDA
`13.0`, compute capability 12.0, FP32 public/FP16 internal, TF32 bật,
`torch.compile(mode="max-autotune")`. Command vẫn dùng GPU project quy định:
`CUDA_VISIBLE_DEVICES=1`, bên trong process là `cuda:0`. Tại lượt 2026-08-29,
`nvidia-smi` map index 1 tới PCI `00000000:81:00.0`, khác PCI `41:00.0` ghi ở
inventory cũ; cả hai implementations trong mỗi pair vẫn chạy cùng device idle.

Full strict accuracy matrix dùng warmup/repeats/rounds `1/1/1` chỉ để gate,
không dùng timing làm performance claim:

```bash
CUDA_VISIBLE_DEVICES=1 python -m tools.matrix_runner \
  --impl v11 --shape-ids 1,2,3,4,5,6,7,8,9,10,11,12,13 \
  --device cuda:0 --dtype float32 --accuracy-trials 5 \
  --warmup 1 --repeats 1 --benchmark-rounds 1 \
  --compile-user --compile-mode max-autotune
```

V11 PASS #1–#13, failed `0`; worst max abs `0.00179082` ở #7, thấp hơn worst
V8.1 matrix cũ `0.00188218`. Artifact:
`runs/benchmarks/matrix_v11_FP32PreGELU_float32_20260829T063100Z.json`.

Paired measurements chạy cả V8.1→V11 và V11→V8.1. #7/#10 dùng accuracy 10,
warmup/repeats/rounds `50/200/5`, profile `20/10`; #6 dùng accuracy 5,
`10/20/3`, profile `10/5`:

```bash
CUDA_VISIBLE_DEVICES=1 python -m tools.profile_models \
  --impl v8.1 v11 --shape-id 7 --accuracy-trials 10 \
  --warmup 50 --repeats 200 --benchmark-rounds 5 \
  --profile-warmup 20 --profile-iterations 10 \
  --compile-user --compile-mode max-autotune
```

| Shape | V11 max-abs so với V8.1 | V11 host latency Δ, hai order | V11 raw GPU Δ, hai order | Kết luận |
|---:|---:|---:|---:|---|
| #7 | `-4.85%` | `+1.47% / +2.77%` | `+0.74% / +0.10%` | Accuracy tốt hơn nhưng latency regress nhỏ |
| #10 | `-1.19%` | `-1.50% / -1.50%` | `-1.17% / -1.11%` | Accuracy và latency cùng tốt hơn |
| #6 | `-13.01%` | `+0.10% / +0.02%` | `+0.003% / -0.064%` | Accuracy tốt hơn, latency trung tính |

Mỗi implementation vẫn có 29 GPU kernels, 2 memory events, 21 Triton events và
một CUDA Graph launch/forward. Xét thuần Pareto, V11 thắng #6/#10 nhưng regress
nhẹ ở #7. Project owner vẫn promote force-all V11 làm main để ưu tiên accuracy
margin và một implementation thống nhất; regression #7 là trade-off được chấp
nhận, không bị trình bày như speedup. Representative artifacts:
`runs/profiles/profile_shape07_20260829T062511Z.json`,
`runs/profiles/profile_shape07_20260829T062542Z.json`,
`runs/profiles/profile_shape10_20260829T062618Z.json`,
`runs/profiles/profile_shape10_20260829T062708Z.json`,
`runs/profiles/profile_shape06_20260829T062800Z.json` và
`runs/profiles/profile_shape06_20260829T062926Z.json`.

### S5.14 — V12 FP32 FFN-out projection output

**Trạng thái:** Implemented và local validated; CUDA #7 accuracy PASS cho V12
FFN-only, clean GPU performance pending do concurrent workload.

**Giả thuyết:** V11 dùng activation/weight/bias FP16 và FP32 GEMM accumulation
cho FFN-out, nhưng `F.linear` vẫn store kết quả FP16 rồi mới cast lên FP32 trước
residual add. Cho GEMM store accumulator trực tiếp thành FP32 có thể bỏ lần
round FP16 ở cuối mỗi FFN branch và tăng accuracy margin. Đây không phải giả
thuyết bỏ kernel cast: max-autotune profile đã cho thấy Inductor fuse cast hiện
tại với residual/LayerNorm. FP32 intermediate còn tăng output traffic, nên
latency dự kiến chỉ trung tính hoặc regress nhỏ.

**Phạm vi V12a:** Kế thừa nguyên V11 và chỉ thay FFN-out projection trên nhánh
FP32-public inference. Hidden, weight và bias cache vẫn FP16; GEMM dùng
`torch.mm(..., out_dtype=torch.float32)` trên CUDA, sau đó cộng chính cached
FP16 bias đã promote lên FP32. Attention output projection, V11 FP32 pre-GELU,
LayerNorm/residual, state dict, cache lifecycle, training và non-FP32 fallback
không đổi. CPU fallback nhân các giá trị FP16 đã quantize sau khi promote FP32
để mô phỏng cùng precision boundary; CPU timing không phải performance claim.

**Gate:** Chạy syntax, strict state-dict, causal/non-causal × padding/no-padding,
training/FP16/BF16 fallback và `torch.compile` smoke trước. Trên RTX 5090,
accuracy-gate #7/#6/#10 trước paired V11/V12 hai implementation orders. Chỉ mở
rộng sang attention out-projection hoặc cả hai boundary nếu V12a pass strict và
performance loss đủ nhỏ; chỉ benchmark row đã PASS.

**Local result:** PyTorch `2.12.1`, CPU. Cả V12/V12.1/V12.2 PASS syntax,
state-dict/cache, causal/non-causal × padding/no-padding, `valid_token_mask=None`
coverage và compile smoke; V12 còn PASS training/FP16/BF16 fallback exact, hai
class phụ kế thừa cùng fallback. Paired 10 seed dùng cùng model/input cho:

| Shape | Variant | Mean abs | Worst max abs | Worst normalized OR risk |
|---:|---|---:|---:|---:|
| #7 | V11 | 0.00018158 | 0.00186644 | 0.759462 |
| #7 | V12 FFN-only | 0.00016808 | 0.00158411 | 0.707284 |
| #7 | V12.1 attention-only | 0.00017533 | 0.00170401 | 0.737809 |
| #7 | V12.2 both | **0.00016120** | **0.00152946** | **0.692520** |
| #10 | V11 | 0.00016430 | 0.00174332 | 0.696029 |
| #10 | V12 FFN-only | 0.00015185 | 0.00156116 | 0.695936 |
| #10 | V12.1 attention-only | 0.00016094 | **0.00145376** | **0.661042** |
| #10 | V12.2 both | **0.00014818** | 0.00146723 | 0.663495 |

V12 FFN-only còn PASS CUDA max-autotune official #7 5/5 với max abs
`0.00138396`. Timing của cùng smoke bị loại vì một `aigc_detector` workload
khác đang dùng GPU vật lý #1; paired latency chỉ chạy lại sau khi GPU idle.

### S5.15 — V13 INT8 FFN-in accuracy probe

**Trạng thái:** Rejected before GPU/kernel work; numerical probe fail strict
accuracy ngay trên official shape #2, không có performance claim.

**Giả thuyết:** Full/per-scope FP8 trước đây fail mạnh vì E4M3 chỉ có ba bit
mantissa và các recipe per-tensor/MXFP8 không giữ đủ độ phân giải. Symmetric
INT8 có nhiều mức hữu hiệu hơn trong một dynamic range đã chọn, nên một scope
FFN-in hẹp có thể dùng ít error budget hơn nếu weight được quantize riêng theo
output channel và activation được quantize động theo từng token row.

**Phạm vi V13a:** Kế thừa nguyên V11 và chỉ thay FFN-in trên public-FP32
inference path. LayerNorm output vẫn round FP16 như V11, sau đó activation dùng
dynamic symmetric per-token scale; FFN-in weight được cache INT8 với symmetric
per-output-channel scale và refresh sau `load_state_dict()`/device move. Dot
product được mô phỏng bằng các integer values, dequantize về FP32, cộng cached
FP16 bias đã promote rồi chạy exact erf-GELU ở FP32; GELU output vẫn round FP16
cho FFN-out. Attention, FFN-out, residual/LayerNorm, state dict, training và
non-FP32 fallback giữ nguyên V11.

Probe còn hỗ trợ hai control qua `TECHJAM_INT8_PROBE_MODE=w8|a8|w8a8` để tách
weight và activation error; `TECHJAM_INT8_PROBE_LAYERS=all|0,2,...` cô lập
scope theo zero-based layer index. Các mode đều là fake-quant numerical
simulation; không được dùng latency của PyTorch emulation làm bằng chứng INT8
speedup.

**Gate:** Chạy syntax/state-dict/cache và causal/non-causal × padding/no-padding
local trước. Strict official #2/#7/#10 là canary nhiều seed; chỉ nếu W8A8 pass
mới mở rộng #6/#8 và viết Triton/CUTLASS INT8 kernel. Nếu W8 pass nhưng W8A8
fail, ghi weight-only là accuracy ceiling chứ không claim Tensor Core speedup.
Mọi performance benchmark vẫn bị chặn cho tới khi có kernel thật và cùng
candidate pass accuracy.

**Local structural validation:** PyTorch `2.12.1`, CPU. `py_compile`, strict
state-dict equality, INT8/FP32 cache dtype, cache refresh sau `load_state_dict()`
và `.to(dtype=float32)`, training fallback cùng BF16 public-input fallback đều
PASS. Diagnostic causal/non-causal × padding/no-padding chạy đúng graph nhưng
cả ba quantization modes đều fail numerical gate; đây không phải performance
workload.

**Official shape #2 canary:** CPU, đúng `B1/S128/D128/H4/FFN128/L4`, causal,
no padding, seed `1234..1238`, strict comparator. Mỗi command dùng
`warmup/repeats/rounds=0/1/1`; do accuracy fail, harness tự skip benchmark:

```bash
TECHJAM_INT8_PROBE_MODE=w8a8 python3 v13_INT8FFNProbe.py \
  --device cpu --dtype float32 \
  --batch-size 1 --seq-len 128 --d-model 128 \
  --heads 4 --ffn-dim 128 --layers 4 --causal \
  --accuracy-trials 5 --warmup 0 --repeats 1 --benchmark-rounds 1
```

| Scope qua 4 layers | Max abs | Failed elements | Gate |
|---|---:|---:|---|
| W8 weight-only | `0.00738502` | `1,659/81,920` | FAIL |
| A8 activation-only | `0.0122206` | `4,574/81,920` | FAIL |
| W8A8 | `0.0198889` | `5,825/81,920` | FAIL |

Weight-only đã fail nên lỗi không chỉ đến từ dynamic activation scaling. Để
kiểm tra scope tối thiểu, đặt `TECHJAM_INT8_PROBE_LAYERS=3` và chỉ quantize
FFN-in của layer cuối. W8 vẫn FAIL `69/81,920`, `max_abs=0.00381088`; W8A8
FAIL `1,332/81,920`, `max_abs=0.00912209`. Sweep tạm từng layer `0..3` cũng
không có row PASS cho W8, A8 hoặc W8A8.

**Kết luận:** Standard symmetric INT8 FFN-in không có accuracy-feasible scope
đủ để biện minh kernel work trên model này. Không chạy #7/#10/#6/#8 hoặc GPU
performance sau khi #2 canary fail. Chỉ mở lại INT8 với recipe materially khác
như outlier routing, quantization-aware correction hay calibration có evidence;
groupwise partial-GEMM không được mặc định là nhanh vì scale theo K-group phá
một GEMM INT8 đơn giản và thêm dequant/reduction overhead.

### S5.16 — V14 memory-bounded batch chunking cho official shape #14

**Trạng thái:** Strict shape-#14 accuracy PASS; optimized-only memory/latency
diagnostic complete. Main promotion và paired baseline speedup không có vì
reference harness không thực thi được ở shape này.

**Giả thuyết:** Shape #14 có input FP32 `[32,100000,1024]` chiếm `12.207 GiB`.
V11 tạo packed QKV FP16 cho cả batch, cần thêm `18.311 GiB`, nên không thể chạy
trên RTX 5090 32 GiB dù Flash Attention không materialize score matrix. Batch
samples hoàn toàn độc lập; chạy một sample qua đủ hai layer rồi ghi vào output
preallocated giữ nguyên semantics nhưng hạ live packed-QKV xuống `0.572 GiB`.

**Candidate:** `v14_BatchChunked.py` kế thừa V11 và chỉ dispatch khi config/input
khớp chính xác official #14, FP32 inference, causal. Chunk size ban đầu là `1`;
mọi shape, dtype và training branch khác fallback nguyên V11. Không thay
baseline, comparator, tolerance hoặc workload.

**GPU probes trước implementation:** RTX 5090 vật lý index `1`, PyTorch
`2.13.0+cu130`, CUDA `13.0`, no padding. V11 optimized-only đúng full input #14
OOM khi cố cấp packed QKV `18.31 GiB`; peak allocated trước lỗi là
`30.588 GiB`. Probe `B=1/S=100000/D=1024/H=16/L=2/FFN=1024` chạy thành công,
peak `2.964 GiB`, first-call wall time `7.506 s` gồm Triton autotune. Đây chỉ là
memory/backend diagnostic, chưa phải accuracy hay performance result.

**Gate:** (1) syntax/state-dict/fallback local; (2) full optimized-only #14
forward và peak memory trên GPU; (3) strict accuracy bằng reference memory-safe
không đổi công thức/comparator; (4) chỉ benchmark sau accuracy PASS. Baseline
harness hiện materialize FP32 score `[32,16,100000,100000]` khoảng `18.6 TiB`,
nên direct harness OOM không được diễn giải là candidate accuracy failure.

**Kết quả GPU ngày 2026-08-30:** cùng RTX 5090 vật lý index `1`, driver
`595.58.03`, PyTorch `2.13.0+cu130`, CUDA `13.0`, FP32 public, TF32 bật, causal,
no padding, seed `1234`. Full candidate forward đúng output
`[32,100000,1024]`, peak `28.526 GiB`, first-call wall time `8.974 s` gồm
autotune/setup. Strict validator dùng cùng official input/weights, chia reference
Q thành block 256 và compare theo token block; local reduced-shape check giữa
original baseline và blocked reference lệch tối đa `3.576e-7`, strict failed
`0`, xác nhận schedule mới không đổi gate.

```bash
CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m tools.shape14.accuracy \
  --device cuda:0 --batch-limit 32 --query-chunk 256
```

Full accuracy PASS `0/3,276,800,000` failed elements, max abs `0.000831008`,
mean abs `6.56362e-05`, elapsed `327.725 s`, peak `21.133 GiB`. Max relative
`1.50101e+08` đến từ reference gần/đúng zero; mọi element đó vẫn pass nhánh
absolute của comparator OR.

Sau accuracy gate, optimized-only CUDA Event diagnostic dùng một warmup, năm
repeat và giải phóng output 12.21 GiB giữa các call ngoài vùng timing:

```bash
CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m tools.shape14.optimized_benchmark \
  --device cuda:0 --warmup 1 --repeats 5
```

Median `6683.9873 ms`, mean `6682.2406 ms`, p90 `6700.7756 ms`, min
`6656.8384 ms`, throughput `478,756.15 token/s`, timed peak `27.000 GiB`.
Baseline latency và speedup là **N/A**, không phải zero: original baseline cần
score tensor khoảng `18.6 TiB`. Vì vậy số trên là exact-shape optimized-only
diagnostic, không phải paired official performance claim.

### S5.17 — V14.1 merge V11/V14 bằng cutoff theo sequence length

**Trạng thái:** Implemented và từng promoted theo D-030; nay là parent/rollback
dưới V15 theo D-031. Local correctness/compile gate complete; GPU evidence kế
thừa V11/V14 vì arithmetic/chunk body không đổi.
Fresh V14.1 GPU rerun deferred do GPU vật lý index 1 đang chạy workload khác.

**Giả thuyết:** Official #1–#13 chỉ có `S <= 1024`, trong khi #14 có
`S=100000`. Dùng cutoff `S >= 8192` bắt large-sequence memory case mà không đổi
arithmetic path của 13 shape cũ. Batch chunking tổng quát theo `S` vẫn exact vì
model không có phép toán giao tiếp giữa các batch sample.

**Implementation:** `v14_1_BatchChunked.py` kế thừa V11. FP32 eval, `B > 1` và
`S >= 8192` chạy chunk size 1; training, non-FP32, `B=1` và dưới cutoff gọi
thẳng V11. Helper chunked dùng `torch.compiler.disable` để tránh graph capture
unroll loop #14. `main.py` và aliases `main`/`best` được promote sang V14.1;
V11/V14 vẫn giữ nguyên làm rollback.

**Local gate:** PyTorch `2.12.1`, causal/non-causal × no-mask/prefix-padding.
Dưới cutoff V14.1 và V11 bitwise-identical; khi cưỡng bức cutoff để chạy chunk
path cả bốn branch cũng bitwise-identical (`max_abs=0`). Meta boundary check
dispatch `S=8191` về V11 và `S=8192` sang chunk. `state_dict` keys bằng nhau;
training/BF16/B=1 fallback đúng. `torch.compile(backend="eager")` PASS cho cả
fallback và chunk branch với `max_abs=0`. Actual CPU Inductor
`mode="reduce-overhead"` cũng strict PASS: fallback max abs `1.19209e-7`,
chunk max abs `0`, failed `0` cho cả hai.

**GPU gate yêu cầu:** (1) paired V11/V14.1 canary trên official #2/#12 để phát
hiện regression nhánh nhỏ; (2) full #14 forward khi bọc `torch.compile`; (3)
minimum official smoke từ `main.py`; chỉ ghi latency khi GPU vật lý index 1
idle. Full strict #14 evidence S5.16 được giữ vì V14.1 gọi cùng V11 per-sample
arithmetic và chỉ thay predicate/compile boundary.

**Remote deployment check:** ba file runtime/harness mới nhất đã được copy sang
RTX 5090 host và SHA-256 khớp local (`v14_1_BatchChunked.py` bắt đầu
`6b67f7d5...`, shape-14 benchmark `dc69b393...`, `main.py` `cea8bbf0...`).
PyTorch `2.13.0+cu130` remote `py_compile` PASS. Fresh timing không được chạy
khi GPU vật lý index 1 đang bị job training khác giữ `12,457 MiB`,
`85–96%` utilization; không thay bằng GPU index 0 và không ghi số nhiễu.

Evidence GPU hiện hữu vẫn áp dụng theo cấu trúc: official #1–#13 gọi trực tiếp
nguyên V11 path đã PASS/benchmark; #14 gọi cùng loop chunk-size-1 và V11
per-sample body của V14 đã PASS `0/3.2768B`, peak `28.526 GiB`, optimized-only
median `6683.9873 ms`. Các số này được giữ nhãn V11/V14 predecessor, không giả
là fresh V14.1 measurement. Fresh compiled #14 rerun vẫn là pre-submission
canary nên làm khi GPU #1 idle.

## 9. S6 — Whole-model compile, FFN và LayerNorm fusion

**Trạng thái:** V4.3 `max-autotune` PASS official #1–#13; shape #14 pending.

### Giả thuyết

V3.1 eager vẫn dispatch nhiều operator nhỏ quanh GEMM/SDPA. Whole-model `torch.compile` cho phép TorchInductor tối ưu graph và sinh fused kernels; mode `reduce-overhead` còn yêu cầu cấu hình giảm Python/kernel-launch overhead bằng CUDA Graph khi graph đủ điều kiện. Standalone custom LayerNorm chỉ được xem xét sau khi compiled trace chứng minh phần reduction/memory traffic đó vẫn là bottleneck.

### Official shape #1 compile ablation đã đo

Hai command dùng cùng RTX 5090 vật lý index `1`, PyTorch `2.13.0+cu130`, CUDA `13.0`, FP32, TF32 bật, seed `1234`, accuracy trials `3`, warmup `50`, repeats `200`, rounds `5`; chỉ khác `--compile-user --compile-mode reduce-overhead`.

```bash
CUDA_VISIBLE_DEVICES=1 python v3_1_CausalMask.py \
  --device cuda:0 --dtype float32 \
  --batch-size 64 --seq-len 128 --d-model 128 \
  --heads 4 --ffn-dim 128 --layers 4 --causal \
  --accuracy-trials 3 --warmup 50 \
  --repeats 200 --benchmark-rounds 5

CUDA_VISIBLE_DEVICES=1 python v3_1_CausalMask.py \
  --device cuda:0 --dtype float32 \
  --batch-size 64 --seq-len 128 --d-model 128 \
  --heads 4 --ffn-dim 128 --layers 4 --causal \
  --accuracy-trials 3 --warmup 50 \
  --repeats 200 --benchmark-rounds 5 \
  --compile-user --compile-mode reduce-overhead
```

| V3.1 execution | Accuracy | Baseline median / p90 | Optimized median / p90 | Throughput | Speedup |
|---|---|---:|---:|---:|---:|
| Eager | PASS, max_abs=`0.00105309`, failed=0/3,145,728 | 1.0806 / 1.1374 ms | 0.5118 / 0.5371 ms | 16,007,004 token/s | 2.112x |
| Compile `reduce-overhead` | PASS, max_abs=`0.00105298`, failed=0/3,145,728 | 1.0786 / 1.1175 ms | **0.3169 / 0.3190 ms** | 25,853,362 token/s | **3.404x** |

Baseline giữa hai process lệch `0.19%`, đủ gần để coi đây là paired environment check. Compile giảm optimized median `38.1%`, tương đương tăng throughput `61.5%` so với v3.1 eager. Compile/setup time xảy ra trước timed steady-state region và chưa được tính vào latency.

Kết quả này chỉ chứng minh whole-model compile có lợi; chưa chứng minh gain đến từ LayerNorm, fusion hay CUDA Graph riêng lẻ. Attribution tiếp theo bắt buộc gồm:

1. Chạy cùng cấu hình với `--compile-mode default` để tách phần tăng thêm của `reduce-overhead`.
2. Dùng compile-aware `tools/profile_models.py` để so raw GPU device/kernel count, `Torch-Compiled Region`, Triton launches, CUDA Graph launches, memory và Chrome trace.
3. Chỉ thử custom Triton residual + LayerNorm nếu compiled trace vẫn cho thấy vùng này đủ lớn.

Candidate còn lại:

- Triton kernel cho residual + LayerNorm.
- Fused Linear + activation khi backend hỗ trợ.
- Giảm intermediate allocation và memory round trip.

Với model dimension lớn, GEMM có thể chiếm ưu thế và custom elementwise kernel không tạo speedup đáng kể; mọi candidate phải đo theo đúng official shape.

### V4.3 max-autotune matrix #1–#13

```bash
TORCH_LOGS=perf_hints CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m tools.matrix_runner \
  --impl v4.3 \
  --shape-ids 1,2,3,4,5,6,7,8,9,10,11,12,13 \
  --compile-user --compile-mode max-autotune
```

Cùng RTX 5090/PyTorch `2.13.0+cu130`, mọi row PASS. Geomean speedup tăng từ
`8.5186x` của reduce-overhead lên `9.5266x` (`+11.83%`); geomean optimized
latency giảm `9.73%`. Shape #8 còn `2.5400 ms`, shape #13 `1.1096 ms`.

Artifact:
`runs/benchmarks/matrix_v4_3_Flash_float32_20260828T124033Z.json`.

## 10. S7 — Custom attention kernel

**Trạng thái:** Idea, chi phí cao.

Chỉ triển khai sau khi profiler chứng minh attention là bottleneck và SDPA không đủ tốt cho shape mục tiêu. Phải hỗ trợ causal/padding semantics và kiểm tra numerical stability, không chỉ kernel throughput riêng lẻ.

## 11. Ma trận benchmark bắt buộc

Performance matrix gồm đúng 14 rows ở Appendix của `STATEMENT.md`; không thêm, giảm hoặc sửa `B`, `S`, `D`, `H`, số layer, FFN dimension hay causal flag. Bảng chính thức không chỉ định padding nên dùng `padding_ratio=0` cho performance cho đến khi có clarification mới.

Correctness/debug vẫn có thể mở rộng dtype, padding, seed và input scale. Với mỗi official performance case, lưu shape ID, pass/fail, worst error, median, p90, throughput và speedup.

## 12. Official matrix runner

**Trạng thái:** Implemented; local và GPU smoke PASS, full GPU matrix pending.

**File:** `tools/matrix_runner.py`.

Runner phải chứa đúng 14 rows từ Appendix, gọi từng benchmark trong subprocess riêng để một case OOM/timeout không làm mất toàn bộ matrix, và lưu kết quả tăng dần ra JSON/CSV. CLI nhận file implementation qua `--impl` (đường dẫn tương đối hoặc tuyệt đối), dtype, measurement settings và subset shape IDs để rerun case lỗi; mặc định chạy toàn bộ 14 shapes. Các alias `v1`, `v2`, `v3`, `v3.1` vẫn được giữ để tương thích.

```bash
CUDA_VISIBLE_DEVICES=1 python -m tools.matrix_runner --impl v3.1
```

Output mặc định nằm trong `runs/benchmarks/matrix_<impl>_<dtype>_<timestamp>.json` và `.csv`. Runner trả exit code `1` nếu có ít nhất một case `ACCURACY_FAIL`, `ERROR` hoặc `TIMEOUT`, nhưng vẫn chạy tiếp và giữ kết quả các case khác.

Bảng tổng kết terminal có cột `max_abs` lấy từ accuracy summary để error margin nằm cạnh latency/speedup. Đây là số diagnostic; status vẫn do strict elementwise `relative < 0.02 OR absolute < 0.002` quyết định, và case không sinh accuracy summary hiển thị `-`.

Local smoke trên official shape #2 và GPU smoke trên official shape #12 đã PASS; parser đọc đúng accuracy, baseline/optimized median, p90, throughput và speedup. Cả hai dùng measurement tối thiểu chỉ để kiểm tra runner, không phải benchmark để báo cáo.

## 13. Reusable PyTorch profiler CLI

**Trạng thái:** Eager và compiled profiler GPU validated trên official shape #1; default-mode/exported Chrome trace còn pending.

**File:** `tools/profile_models.py`.

Profiler chạy mỗi implementation trong process riêng theo thứ tự bắt buộc: accuracy gate → alternating baseline/optimized timing → optimized-path operator profile. Input profile dùng chính `generate_random_case()` của benchmark harness, gồm cả all-valid `valid_token_mask` khi `padding_ratio=0`; không thay bằng `None` vì hai đường mask có thể tạo operator graph khác nhau.

```bash
CUDA_VISIBLE_DEVICES=1 python -m tools.profile_models \
  --impl v1 v2 v3 --shape-id 1
```

Eager terminal in end-to-end latency, raw GPU/runtime evidence, operator category breakdown, non-overlapping model-stage breakdown, top ATen operators và top GPU device events. Stage scopes chỉ được gắn trong eager profiler process, không đi vào benchmark path. Các stage gồm pre-attention LayerNorm, packed/separate QKV projection, view/reshape, attention core, output projection, residual, pre-FFN LayerNorm, FFN in/GELU/out, masking/padding, copy và final LayerNorm.

Compiled path dùng:

```bash
TORCH_LOGS=perf_hints CUDA_VISIBLE_DEVICES=1 python -m tools.profile_models \
  --impl v3.1 --shape-id 1 \
  --compile-user --compile-mode reduce-overhead \
  --export-traces --record-shapes
```

Compiled profiler không gắn eager stage scopes vì Inductor có thể fuse nhiều ATen operator vào cùng Triton kernel và instrumentation muộn có thể đổi/recompile graph. Thay vào đó JSON/terminal lưu raw GPU device time, kernel/memory-event count, Triton event/launch count, compiled-region count, CUDA Graph launch API, steady/peak CUDA allocation, top device events và compile mode/options. Child compiled bật `TORCHINDUCTOR_UNIQUE_KERNEL_NAMES=1` để trace có tên kernel dễ audit.

Latency dùng CUDA Event; eager operator/stage self time dùng PyTorch Profiler/Kineto và chỉ tổng hợp row `aten::*` để tránh double-count raw CUDA kernel rows. Compiled GPU evidence chỉ tổng hợp raw CUDA device events nên cũng không cộng lại với ATen rows. `attention_core` giữ fused SDPA nguyên vẹn: `QKᵀ`, scale, causal/key mask, softmax và `probabilities @ V` không thể đo tách chính xác mà không thay kernel bằng explicit diagnostic khác. JSON mặc định nằm trong `runs/profiles/`; bật Chrome trace khi cần mở timeline trong Perfetto:

```bash
CUDA_VISIBLE_DEVICES=1 python -m tools.profile_models \
  --impl v1 v2 v3 --shape-id 1 --export-traces --record-shapes
```

Các profiler trace có thể lớn và `runs/profiles/` không được commit mặc định.

## 14. S8.1 — V15 direct-layout QKV cho attention-heavy shape #13

**Trạng thái:** Accepted và promoted theo D-031. Local/CUDA correctness PASS;
paired max-autotune #13 giữ end-to-end và raw-device gain ở cả hai orders.

**Profiler evidence:** Compiled V4.3 shape #13 có median `1.1453 ms` và raw
device time `1.1474 ms`. Bốn Flash Attention calls chiếm `0.4211 ms`; các
Tensor Core GEMM chiếm khoảng `0.4012 ms`. Inductor đã sinh riêng các
Triton kernel fuse residual/mask/LayerNorm/cast, nên standalone LayerNorm
kernel sẽ lặp lại negative ablation V7. Phần packed QKV hiện tại là
view không copy nhưng Q/K/V có sequence stride `3D`, vì ba projection
interleave theo từng token.

**Giả thuyết:** Một Triton FP16 QKV GEMM với FP32 accumulator có thể
ghi epilogue thẳng ra storage `[3,B,H,S,Dh]`. Cách này bỏ packed
`[B,S,3D]` và cấp cho Flash ba tensor `[B,H,S,Dh]` contiguous, giữ nguyên
weights/bias, attention scale, exact Flash SDPA và các precision boundary
khác của V14.1/V11. Gain phải đến từ trade-off giữa direct layout
và chất lượng custom GEMM; không suy từ việc bỏ view vì view cũ
vốn không materialize copy.

**Phạm vi ablation:** `v15_DirectQKVLayout.py` chỉ dispatch cho exact official
shape #13 (`B64/S1024/D128/H4/L4/FFN128`, causal, FP32 eval). Mọi config,
training/dtype khác và large-sequence #14 fallback nguyên V14.1. CPU/no-
Triton fallback dùng `F.linear` rồi materialize cùng direct layout để test
semantics; chỉ CUDA Triton path mới là performance candidate.

**Gate:** (1) syntax, strict state dict/cache, branch fallback; (2) local
causal/no-mask và prefix-padding equivalence; (3) CUDA official #13 strict
accuracy nhiều trial; (4) paired V14.1/V15 theo cả hai implementation orders
với `torch.compile(mode="max-autotune")`; (5) profiler so QKV/Flash raw device
time, kernel count và peak memory. Chỉ promote nếu whole-model gain giữ dấu
và không làm mất correctness margin.

**Implementation:** Custom op `techjam::direct_qkv_projection` dùng Triton
autotune trên sáu tile configs. Input/packed weight/bias là FP16, `tl.dot`
accumulate FP32, bias được cộng trước một lần store FP16. Store
epilogue map logical column `n` thành `(qkv,head,head_channel)` và token row
`m` thành `(batch,sequence)`, tạo storage `[3,B,H,S,Dh]` không cần kernel
transpose. Candidate kế thừa V14.1; dispatch key là exact #13 và mọi
nhánh khác gọi nguyên inherited path.

**Local gate (PyTorch 2.12.1, CPU):** `py_compile`, aliases/list-shapes, strict
state-dict key equality, cache refresh exact sau strict `load_state_dict()` và
`.to(dtype=float32)` (packed cache vẫn FP16), training fallback, BF16 fallback và non-target
V14.1 fallback đều PASS. Direct-layout projection khớp packed-projection
reference bitwise và trả tensor contiguous đúng `[3,B,H,S,Dh]`. Exact-#13
dispatch chạy diagnostic `B2/S17/D128/H4/L4/FFN128` trên ba seed, có/không
prefix padding, strict failed `0/26,112`, max abs `0.00111527`. V14.1/V15
CPU outputs bitwise-identical cho hai mask branches. `torch.compile(backend=
"eager")` khớp exact; CPU Inductor `reduce-overhead` strict failed `0/4,352`,
max abs so với eager `0.000270784`.

Stride diagnostic `B2/S11/D128/H4`: packed Q view có stride
`(4224,32,384,1)`, non-contiguous; direct Q có stride
`(1408,352,32,1)`, contiguous, trong khi values bitwise-equal. Trên official
#13, sequence stride tương ứng đổi từ `384` xuống `32`; đây là
biến V15 thật sự thay đổi, không phải bỏ một copy đã tồn tại.

**Remote structural gate:** file SHA-256 bắt đầu `65efad80...`; remote
PyTorch `2.13.0+cu130`/Triton `3.7.1` import và `py_compile` PASS. Lượt đầu đã
defer đúng lúc GPU #1 bị process khác giữ `12,457 MiB`, `91–96%` utilization;
CUDA accuracy/timing dưới đây chỉ chạy sau khi GPU trở lại idle.

Cross-compile không launch GPU bằng Triton `ASTSource` + explicit
`GPUTarget("cuda", 120, 32)` PASS cả sáu autotune configs. PTX/cubin sizes:
`BM64/BN64/BK32/W4 = 41,235/60,304`, `128/64/32/W4 = 63,184/91,216`,
`64/128/32/W8 = 38,179/57,176`, `128/128/32/W8 = 57,147/89,104`,
`64/128/64/W8 = 50,052/75,024`, `128/128/64/W8 = 74,847/112,992`
bytes; mọi config dùng bốn stages. Kernel name là
`_direct_qkv_projection_kernel`. Gate này chứng minh source lowering/codegen
SM120 hợp lệ, nhưng không thay thế CUDA execution, accuracy hay timing.
Remote `tools/matrix_runner.py`/V15 hashes khớp local và alias `v15` resolve đúng.

**CUDA semantic/branch gates:** Custom projection trên GPU trả contiguous
`[3,2,4,17,32]`, Q stride `(2176,544,32,1)`, và khớp packed `F.linear`
bitwise (`max_abs=0`). Eager lẫn compiled model diagnostic đều PASS
`0/4,352`, max abs `0.000856102`. Official #13 với prefix padding 25% PASS ba
trial `0/25,165,824`, max abs `0.00147235`. Non-causal + prefix-padding
diagnostic chạy inherited V14.1 path và PASS `0/13,056`, max abs `0.00048244`.

**Official #13 strict gate:**

```bash
TORCH_LOGS=perf_hints CUDA_VISIBLE_DEVICES=1 .venv/bin/python \
  v15_DirectQKVLayout.py --device cuda:0 --dtype float32 \
  --batch-size 64 --seq-len 1024 --d-model 128 --heads 4 \
  --ffn-dim 128 --layers 4 --causal --accuracy-trials 5 \
  --warmup 1 --repeats 1 --benchmark-rounds 1 \
  --compile-user --compile-mode max-autotune
```

Năm trial PASS `0/41,943,040`, worst max abs `0.00147235`. Latency optimized
một repeat ngay sau compile là warmup artifact và không được dùng làm claim.

**Paired performance gate dài:** Cùng RTX 5090, FP32 public/FP16 internal,
seed `1234`, `accuracy=5`, warmup/repeats/rounds `50/200/5`, profile warmup/
iterations `50/20`, max-autotune:

| Order | V14.1 median / p90 | V15 median / p90 | End-to-end Δ | V14.1 raw GPU | V15 raw GPU | Raw Δ |
|---|---:|---:|---:|---:|---:|---:|
| V14.1 → V15 | 1.1080 / 1.1166 ms | **1.0971 / 1.1049 ms** | **-0.98%** | 1.0966 ms | **1.0838 ms** | **-1.17%** |
| V15 → V14.1 | 1.1251 / 1.1330 ms | **1.1003 / 1.1085 ms** | **-2.20%** | 1.1062 ms | **1.0867 ms** | **-1.76%** |

Baseline medians là `41.7740–41.7829 ms`; V15 speedup theo process tương ứng
`38.077x` và `37.965x`. Cả hai directions giữ dấu. Short paired gate
`20/100/3` cũng giữ dấu (`-2.37%/-1.70%` end-to-end và
`-1.86%/-1.81%` raw GPU), nhưng bảng dài là evidence promotion chính.

Profiler của cả hai candidate có 29 GPU kernels, 2 memory events, một compiled
region và một CUDA Graph launch. V14.1 có 21 Triton GPU events; V15 có 17.
V15 thay projection region bằng `_direct_qkv_projection_kernel` bốn lần;
Flash time gần trung tính, nên không claim Flash kernel nhanh hơn. Gain đến từ
tổ hợp QKV GEMM + layout/consumer graph.

Artifacts:

- `runs/profiles/profile_shape13_20260830T034815Z.json`
  (V14.1 → V15, gate dài).
- `runs/profiles/profile_shape13_20260830T035041Z.json`
  (V15 → V14.1, gate dài).
- `runs/profiles/profile_shape13_20260830T034539Z.json` và
  `profile_shape13_20260830T034652Z.json` (short paired controls).

**Fallback canaries:** Official #2/#12 chạy V15 alias với
`accuracy=3`, warmup/repeats/rounds `20/100/3`, max-autotune. Cả hai PASS:
#2 max abs `0.000905275`, optimized `0.0781 ms`, speedup `13.106x`; #12 max
abs `0.00127272`, optimized `0.0780 ms`, speedup `13.223x`. Artifact:
`runs/benchmarks/matrix_v15_DirectQKVLayout_float32_20260830T035331Z.json`.

**Kết luận:** D-031 promote V15 qua `main.py`; V14.1 được giữ làm rollback.
Dispatch vẫn exact #13, nên evidence/behavior #14 của V14.1 không đổi. Gain
`0.98–2.20%` đủ giữ dấu trên GPU mục tiêu nhưng cần paired-rerun trên hardware
khác trước khi xem schedule này là portable.

Sau promotion, remote `main.py` SHA-256 `854ddd38...` và V15 SHA-256
`65efad80...`; `py_compile` PASS. `tools/matrix_runner.py --impl main --shape-ids 13`
PASS strict (`max_abs=0.00147235`), xác nhận stable entrypoint chạy V15. Timing
của smoke này bị loại: trong lúc đo xuất hiện process Track 5 chiếm GPU #1 ở
`76%` utilization, làm baseline tăng `~41.8 → 78.2 ms`. Không dùng speedup
`72.993x` của artifact nhiễu trong report; paired idle artifacts phía trên mới
là performance evidence.

## 15. S8.2 — V16 reusable compiled B=1 executor cho shape #14

**Baseline/hypothesis:** V14/V14.1 đã làm shape #14 khả thi bằng vòng lặp eager
32 batch slice độc lập, nhưng annotation `@torch.compiler.disable` hiện loại cả
thân `B=1` khỏi Inductor. Evidence hợp lệ hiện tại là optimized-only median
`6683.9873 ms`, peak allocated `27.000 GiB`; baseline gốc không chạy được vì
score tensor ước tính `18.6 TiB`. V16 sẽ giữ vòng lặp ngoài eager để không
unroll 32 sample/OOM, compile đúng callable `B=1` một lần sau khi model đã
load weight/chuyển device/eval, rồi tái sử dụng callable đó cho mọi slice.
Compilation/autotune không được tính vào latency.

**Phạm vi:** Candidate kế thừa V15 để giữ exact-#13 direct-QKV và toàn bộ
fallback V14.1. Dispatch mới chỉ áp dụng FP32 eval, `B>1`, `S>=8192`; training,
dtype khác, sequence ngắn và lời gọi `B=1` trực tiếp giữ nguyên inherited path.
Compiled executor phải bị invalidate sau `load_state_dict()`, `_apply()`/`.to()`
và thay đổi training mode. Không thay arithmetic, tolerance, workload hay test.

**Rủi ro:** Dynamo có thể graph-break khi compile bound method/custom op;
CUDA Graph có thể không tái sử dụng vì mỗi batch slice có data pointer khác;
compile có thể tăng peak memory vượt 32 GiB hoặc đổi sai số FP16/FP32. Chỉ thêm
static staging buffer nếu profiler chứng minh pointer churn là bottleneck và
memory gate cho phép; không coi compile thành công là speedup.

**Gate:** (1) syntax/import/state-dict/cache lifecycle; (2) local forced-chunk
equivalence có/không mask, training/non-FP32/non-target fallback và
`torch.compile(backend="eager")`; (3) CUDA semantic canary cho compiled B=1;
(4) strict accuracy chính thức #14 bằng reference query-blocked, ít nhất canary
trước rồi đủ 32 batch nếu performance giữ dấu; (5) optimized-only V14.1/V16
cùng GPU #1, seed, warmup/repeats, TF32 và compile mode khi GPU idle. Promote
chỉ khi median giảm có thể lặp lại, peak <32 GiB và strict comparator vẫn PASS.

**Implementation/local gates:** `v16_CompiledBatchExecutor.py` kế thừa V15,
override riêng helper large-sequence. Loop batch vẫn compiler-disabled; compiled
bound method B=1 được cache ngoài module state. Cache invalidate sau strict
`load_state_dict()`, `.to()`/`_apply()` và train/eval transition. Local forced
cutoff B2/S17/D128 causal có/không mask khớp V15 bitwise; training, BF16,
short-sequence và top-level compiled-outer fallbacks PASS. CPU Inductor PASS
strict `0/4,352`, max abs `0.000260085`. `py_compile`, alias `v16` và import
PASS.

**CUDA canaries:** Trên server PyTorch `2.13.0+cu130`, non-official
B2/S257/D128/L2 diagnostic PASS `0/65,792`, max abs `0.000742763`; log Inductor
xác nhận Flash/custom-op lowering và GEMM autotune. Official #14 batch-limit 1
PASS `0/102,400,000`, max abs `0.00068482`, peak `19.607 GiB`.

User chuyển measurement sang Vast.ai RTX 5090 idle: Ubuntu 24.04 container,
driver `595.58.03`, PyTorch `2.11.0+cu128`, CUDA wheel `12.8`, Triton `3.6.0`,
seed `1234`, TF32 bật. Instance có một GPU logic `cuda:0`; `/workspace` không
persistent. Broad repo sync bị loại vì `AGENTS.md` chứa credential của server
cũ; chỉ source `.py` đã audit được chuyển, không chuyển secret.

**Full strict official #14:** Candidate compiled sample executor, query chunk
256, compare-token chunk 2048:

```bash
CUDA_VISIBLE_DEVICES=0 /venv/main/bin/python -m tools.shape14.accuracy \
  --device cuda:0 --impl v16 --batch-limit 32 --query-chunk 256 \
  --compare-token-chunk 2048 --seed 1234 --compile-mode max-autotune
```

PASS `0/3,276,800,000`, max abs `0.000944197`, mean abs `6.56367e-05`, elapsed
`332.308 s`, peak `19.585 GiB`. `max_rel=1.51706e+08` đến từ reference sát 0;
mọi element vẫn pass strict OR comparator bằng absolute branch.

**Optimized-only sandwich:** Cùng exact #14, một warmup/năm repeat, seed/TF32/
cleanup giống nhau. Original baseline và speedup vẫn N/A.

```bash
CUDA_VISIBLE_DEVICES=0 /venv/main/bin/python -m tools.shape14.optimized_benchmark \
  --device cuda:0 --impl v14.1 --warmup 1 --repeats 5 \
  --compile-mode max-autotune
CUDA_VISIBLE_DEVICES=0 /venv/main/bin/python -m tools.shape14.optimized_benchmark \
  --device cuda:0 --impl v16 --warmup 1 --repeats 5 \
  --compile-mode max-autotune
```

| Sequence | Median / p90 | Throughput | Peak |
|---|---:|---:|---:|
| V14.1 trước | 7396.7202 / 7420.3896 ms | 432,624 token/s | 26.977 GiB |
| V16 giữa | **7166.8359 / 7178.9326 ms** | **446,501 token/s** | **24.487 GiB** |
| V14.1 sau | 7435.5688 / 7451.3472 ms | 430,364 token/s | 26.977 GiB |

V16 giảm median `3.11%` so control trước và `3.61%` so control sau; peak giảm
`2.490 GiB` (`9.23%`). Gain giữ dấu trong sandwich nhưng chỉ được claim là
optimized head-to-head #14, không phải speedup so original baseline.
Artifact đầy đủ:
`runs/benchmarks/shape14_v16_v141_sandwich_20260830.json`.

Official #2/#13 fallback canaries PyTorch 2.11 PASS strict, max abs
`0.000905275`/`0.00147235`. Timing `warmup/repeats/rounds=1/1/1` bị loại.
Artifact fallback:
`runs/benchmarks/matrix_v16_CompiledBatchExecutor_float32_20260830T113404Z.json`.

**Kết luận:** D-032 promote V16 qua `main.py`. V15 giữ QKV rollback, V14.1 giữ
eager large-sequence rollback. PyTorch 2.13 mới có correctness canary; cần fresh
V16/V14.1 performance rerun trên stack đó trước khi xem gain portable.

Sau promotion, local/remote SHA-256 khớp: `main.py` bắt đầu `7a59de73...`, V16
bắt đầu `4cfb8755...`; remote `py_compile` PASS. Stable alias `main` chạy
official #2 trên PyTorch 2.11 và PASS strict ba trial, max abs `0.000905275`.
Timing của smoke `warmup/repeats/rounds=1/1/1` bị loại; chỉ dùng để xác nhận
entrypoint đã trỏ V16 và branch nhỏ không bị ảnh hưởng.

## 16. S8.3 — V17 compiled batch-chunk B=2 cho shape #14

**Trạng thái:** Full strict accuracy PASS và benchmarked; không promote vì gain
chỉ `0.30–0.59%`, V16 vẫn là main/control theo D-033.

**Giả thuyết:** V16 hạ timed peak của shape #14 từ `26.977` xuống
`24.487 GiB`, còn khoảng `8.1 GiB` so với RTX 5090 32 GiB. Chạy hai batch
sample trong cùng compiled executor có thể giảm số lần gọi thân Transformer từ
32 xuống 16, tăng kích thước GEMM/Flash workload và amortize launch/dispatcher
overhead. Arithmetic, precision boundary, cutoff `S>=8192`, full output, input,
weights và strict comparator không đổi; biến độc lập duy nhất là large-sequence
batch chunk `1 -> 2`.

**Phạm vi candidate:** Tạo version riêng `v17_CompiledBatch2.py`, kế thừa V16
và giữ V16 làm rollback. Executor chấp nhận chunk batch từ 1 tới 2 để xử lý
đuôi batch lẻ, nhưng official #14 `B=32` chỉ compile/chạy specialization B=2.
Thân executor phải bypass V14.1 batch dispatcher để không recurse khi nhận
`B=2`, đồng thời vẫn dùng đúng V11/V15 arithmetic của shape #14. Cache compile,
invalidation sau `load_state_dict()`/`_apply()`/train-eval và outer
`torch.compiler.disable` giữ nguyên V16.

**Gate:** (1) syntax/import/alias/state-dict/cache lifecycle; (2) local forced
cutoff B=2 có/không mask phải khớp V16/V15, cùng training/non-FP32/short-shape
fallback; (3) accuracy harness phải gọi candidate theo group B=2 rồi compare
từng sample với query-blocked reference, không được vô tình gọi B=1; (4) CUDA
official batch-limit 2 strict canary; (5) chỉ sau canary mới chạy một-repeat
memory/latency diagnostic, gắn nhãn không hợp lệ để promote; (6) nếu B=2 vừa
PASS vừa nhanh hơn và peak an toàn, chạy full 32-batch strict accuracy trước
sandwich V16->V17->V16 cùng seed/TF32/warmup/repeats. Chỉ promote nếu full
accuracy PASS, median giữ dấu và peak nằm dưới VRAM target; không tự động thử
B=4 nếu B=2 đã OOM/regress.

**Implementation/local gates:** `v17_CompiledBatch2.py` kế thừa V16, đặt
`_LARGE_SEQUENCE_BATCH_CHUNK=2` và gọi thẳng V11 arithmetic body bên trong
compiled executor để tránh V14.1 recurse ở `B=2`. Executor vẫn nhận `B=1` cho
đuôi batch lẻ. Accuracy harness được đổi sang gọi candidate theo group size của
executor rồi compare từng sample với query-blocked reference. Local B2/B3
causal có/không mask khớp V16 bitwise; short-sequence, training và BF16
fallback cũng bitwise-equal. Cache lifecycle sau load/apply/train, syntax,
alias/list-shapes và official #2 CPU fallback smoke PASS.

**CUDA gates:** Cùng Vast.ai RTX 5090/PyTorch `2.11.0+cu128` của V16, strict
batch-limit-2 canary chạy đúng header `executor_batch=2` và PASS
`0/204,800,000`, max abs `0.000719786`, peak `21.880 GiB`. Full gate:

```bash
CUDA_VISIBLE_DEVICES=0 /venv/main/bin/python -m tools.shape14.accuracy \
  --device cuda:0 --impl v17 --batch-limit 32 --query-chunk 256 \
  --compare-token-chunk 2048 --seed 1234 --compile-mode max-autotune
```

PASS `0/3,276,800,000`, max abs `0.000944197`, mean abs `6.56366e-05`,
elapsed `333.542 s`, peak `20.348 GiB`. Error profile về cơ bản bằng V16;
executor B=2 không dùng thêm precision budget. Official #2/#13 fallback
canaries cũng PASS strict, max abs `0.000905275`/`0.00147235`; timing `1/1/1`
bị loại.

**Alternating optimized-only benchmark:** Original baseline/speedup tiếp tục
N/A. Mỗi process dùng một warmup, năm repeats, cùng seed/TF32/max-autotune và
cleanup output ngoài timing:

| Sequence | Median / p90 | Throughput | Timed peak |
|---|---:|---:|---:|
| V16 control trước | 7183.8022 / 7202.1914 ms | 445,447 token/s | 24.487 GiB |
| V17 B=2 lần 1 | **7141.3345 / 7151.5054 ms** | 448,096 token/s | 24.487 GiB |
| V16 control sau | 7162.9731 / 7174.2969 ms | 446,742 token/s | 24.487 GiB |
| V17 B=2 lần 2 | **7131.5425 / 7148.6027 ms** | 448,711 token/s | 24.487 GiB |

V17 lần 1 nhanh hơn hai controls `0.59%` và `0.30%`; lần 2 nhanh hơn V16 ngay
trước `0.44%`. Trung bình hai median mỗi implementation là
`7173.3877 → 7136.4385 ms` (`-0.515%`). Dấu tốt được lặp lại nhưng effect dưới
1%, trong khi từng process có drift tăng latency theo repeat. Theo D-033, không
đủ confidence để thay main bằng một specialization B=2 mới; giữ V17 làm
versioned ablation và không thử B=4 trong lượt này. Artifacts:
`runs/benchmarks/shape14_v17_v16_alternating_20260830.json` và
`runs/benchmarks/matrix_v17_CompiledBatch2_float32_20260830T121859Z.json`.

## 17. S8.4 — Profile inner executor shape #14 trước V18

**Trạng thái:** Hoàn tất profiling và backend shootout; attention đã được xác
nhận là bottleneck, chưa có paired official speedup vì baseline #14 không chạy
được trên 32 GiB.

**Giả thuyết:** V17 chỉ giảm median khoảng `0.515%` so với V16 dù số lần gọi
compiled executor giảm một nửa. Điều này gợi ý Python/dispatcher/launch overhead
không còn là phần đủ lớn để tiếp tục tăng batch chunk. Với `S=100000`, causal
attention có độ phức tạp bậc hai theo sequence trong khi QKV, output projection,
FFN và LayerNorm chỉ tuyến tính theo sequence; attention backend nhiều khả năng
chiếm phần lớn device time. Đây mới là giả thuyết, chưa được dùng để chọn V18.

**Phạm vi diagnostic:** Thêm `tools/shape14/profile.py` chỉ cấp phát đúng input của
inner executor (`B=1` cho V16, `B=2` cho V17), gọi trực tiếp
`forward_large_sequence_sample()` sau compile/warmup và thu CUDA Event latency,
raw Kineto CUDA events, runtime launch counts cùng peak allocation. Không cấp
phát full input/output `B=32`, nên artifact phải ghi rõ là **inner-executor
diagnostic**, không phải official full-forward latency hay speedup. B=2 được báo
cả per-call và normalize per-sample để so với B=1.

**Attribution:** Raw device events được nhóm heuristic thành attention, GEMM,
FFN-in/GELU, LayerNorm/residual/mask, memory và other. Tên sự kiện gốc cùng
category rule phải lưu trong JSON; tổng category chỉ là profiler attribution,
không được cộng với ATen self time hoặc CUDA Event wall latency như các phép đo
độc lập. Chạy với `TORCHINDUCTOR_UNIQUE_KERNEL_NAMES=1` để tăng khả năng audit.

**Gate chọn V18:** (1) profiler harness syntax/import và output contract; (2)
V16/V17 chạy cùng exact shape-#14 config, seed, TF32, compile mode và target GPU
idle; (3) attention share phải được xác nhận từ raw device events. Nếu attention
chiếm áp đảo, V18 sẽ là exact-shape backend/kernel shootout trước khi đụng
QKV/FFN/LayerNorm; nếu không, candidate phải nhắm category lớn nhất đã đo.
Không sửa `main.py`, không chạy full strict accuracy mới và không claim speedup
chỉ từ profiler diagnostic này.

**Kết quả profiler:** Trên Vast.ai RTX 5090, PyTorch `2.11.0+cu128`, CUDA
`12.8`, TF32 bật và inner executor exact `S=100000,D=1024,H=16,L=2`, V16 B=1
đạt median `218.4892 ms/call` (`215.8004 ms` raw device). Raw event attribution
cho attention `199.0931 ms`, chiếm **92.258%**; linear projections chỉ
`10.6673 ms` (`4.943%`), FFN `2.2905 ms` (`1.061%`). V17 B=2 đạt
`436.0289 ms/call = 218.0144 ms/sample`; batching chỉ giảm khoảng `0.22%` ở
inner level, khớp whole-forward gain nhỏ trước đó. Artifacts:
`runs/profiles/shape14_inner_v16_20260830T123125Z.json` và
`runs/profiles/shape14_inner_v17_20260830T123147Z.json`.

**Built-in backend sandwich:** Exact V16 inner executor, cùng seed/config,
warmup 1/repeats 15: Flash control trước `217.3188 ms`, cuDNN
`223.6575 ms`, Flash control sau `218.4642 ms`; cuDNN thua Flash
`2.38–2.92%`. Efficient backend khoảng `417.5209 ms`, gần `2x` chậm hơn.
Vì vậy không tạo version mới chỉ để ép một SDPA backend có sẵn.

**External probes:** FlashAttention-4 `4.0.0b28` exact attention đạt
`108.7358 ms` so PyTorch Flash `100.9365 ms` (`0.9283x`, chậm `7.72%`) dù
strict attention output PASS `0/102,400,000`; reject theo performance.
SageAttention source commit `d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5`:
SM120 auto INT8-QK/FP8-PV fail mạnh; recipe INT8-QK per-thread + PV-FP16 với
FP32 accumulation đạt `72.1337 ms` so Flash `100.5045 ms` (`1.3933x`) nhưng
attention-level strict FAIL `94/102,400,000`, nên chỉ được đưa qua full-model
accuracy gate dưới version riêng V18, chưa phải performance claim.

**External-kernel follow-up:** Nếu built-in sweep xác nhận PyTorch Flash thắng
cuDNN/Efficient nhưng attention vẫn trên `90%`, probe tiếp FlashAttention-4
CuTeDSL bằng bản PyPI được pin và ghi version cụ thể. Đây là dependency
experimental trên Vast workspace không persistent; không thêm vào submission
cho tới khi exact `B1/H16/S100000/Dh64` causal kernel chạy được, thắng isolated
và inner-executor timing, rồi V18 qua strict full-model accuracy. SageAttention
FP8/FP4 chỉ đứng sau FA4 vì error budget của model đã loại nhiều low-precision
candidate; kernel throughput không đủ để bỏ qua strict comparator.

## 18. S8.5 — V18 exact-#14 SageAttention PV-FP16 candidate

**Trạng thái:** Rejected ở strict accuracy gate; không benchmark model.

**Evidence đầu vào:** Profiler V16 B=1 xác nhận PyTorch Flash chiếm trên `92%`
raw device time. Built-in Efficient chậm gần `2x`, cuDNN thua Flash trong
sandwich dài và FA4 b28 chậm hơn PyTorch Flash `7.72%` trên exact
`B1/H16/S100000/Dh64`. SageAttention automatic SM120 recipe INT8-QK/FP8-PV
fail mạnh. Recipe chính xác hơn INT8-QK per-thread + PV FP16/FP32 accumulation
trên QKV thật của layer đầu vẫn fail attention-level comparator ở S=1024
`73/1,048,576`, nhưng exact S=100000 chỉ fail `94/102,400,000` và đạt
`72.1337 ms` so PyTorch Flash `100.5045 ms` (`1.3933x`).

**Giả thuyết:** Sai số attention-level còn lại có thể bị attention output
projection, FP32 residual và LayerNorm attenuate đủ để full Transformer vẫn
pass strict contest comparator. Nếu pass, giảm khoảng `28 ms` cho mỗi trong
hai attention layer có upside end-to-end lớn hơn nhiều các ablation dưới 1%
trước đó. Đây chưa phải correctness evidence; direct attention fail không được
đổi thành claim model PASS bằng suy luận.

**Phạm vi V18:** Version riêng kế thừa V16; chỉ exact config #14, causal FP32
eval và CUDA có optional SageAttention mới thay attention core. QKV projection,
interleaved layout, scale, output projection, V11 FFN, residual/LayerNorm,
batch chunk B=1 và compiled executor giữ nguyên. Sage call dùng custom-op wrapper
để Inductor có fake/meta contract; host không có dependency hoặc branch khác
fallback nguyên V16. Source SageAttention pin commit
`d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5`, build SM120 trong overlay Vast
không persistent; không thêm dependency vào stable submission trước promotion.

**Gate:** (1) local syntax/state-dict/fallback/mask semantics; (2) CUDA eager
attention wrapper canary; (3) compiled executor phải capture/run được, hoặc ghi
rõ non-CUDA-Graph configuration nếu library yêu cầu; (4) strict official #14
`batch-limit=1` bằng query-blocked reference trước mọi timing. Nếu canary fail
dù một element, reject V18 và dừng; nếu PASS, chạy batch-limit 2 rồi full 32
strict accuracy, sau đó alternating V16/V18 optimized-only benchmark và memory
gate. Không sửa `main.py` cho tới khi toàn bộ gates pass.

**Implementation/local gate:** `v18_SageAttentionShape14.py` và aliases/harness
được thêm dưới version riêng. Local syntax, `state_dict`, exact-config CPU
fallback và non-target equality PASS. Optional dependency không có trên host
thì branch giữ V16. Fake/meta output stride ban đầu sai vì Sage materialize
contiguous output; đã sửa contract sang contiguous và buộc fresh Inductor cache
để không tái dùng artifact cũ.

**Strict canary và quyết định:** Cùng RTX 5090/PyTorch `2.11.0+cu128`, lệnh
accuracy-only hợp lệ (không compile) là:

```bash
CUDA_VISIBLE_DEVICES=0 /workspace/sageenv/bin/python -m tools.shape14.accuracy \
  --device cuda:0 --impl v18 --batch-limit 1 --query-chunk 256 \
  --compare-token-chunk 2048 --seed 1234 --disable-inner-compile
```

Kết quả **FAIL `1/102,400,000`**, max abs `0.0026415`, mean abs
`7.90562e-05`, elapsed `12.705 s`, peak `21.112 GiB`. Strict `<` nghĩa là chỉ
một outlier cũng đủ reject. Không chạy batch-limit 2/full 32 và không chạy
performance benchmark.

Compiled integration còn fail độc lập: custom-op fresh-cache run tạo output
sai `59,375,874/102,400,000`, max abs `1.08499`. Run này chỉ là integration
debug, không phải accuracy evidence cho thuật toán Sage eager và tuyệt đối
không được dùng làm benchmark. Kết luận giữ V18 như negative ablation; `main.py`
tiếp tục V16. Muốn mở lại phải có recipe QK chính xác hơn **và** wrapper được
`torch.library.opcheck`/compiled-eager equivalence xác nhận trước full gate.

## 19. S8.6 — V19 selective exact-prefix correction cho SageAttention

**Trạng thái:** Diagnostic planned; chưa có candidate/performance result.

**Evidence và giả thuyết:** V18 Sage PV-FP16 nhanh hơn isolated PyTorch Flash
`1.3933x`, nhưng exact attention fail `94/102.4M` và full Transformer B=1 còn
đúng một strict violation. Với causal attention, query ở đầu sequence chỉ có ít
valid keys; quantized QK/softmax có thể tạo outlier tập trung ở một prefix nhỏ.
Nếu toàn bộ violations nằm trong query prefix `P << S`, có thể chạy PyTorch
Flash exact chỉ cho `Q[:P]` với full K/V, rồi overwrite prefix của Sage output.
Chi phí bổ sung là `O(P*S)` thay vì một attention exact `O(S^2)` thứ hai; math
cho corrected rows đúng PyTorch Flash, còn phần suffix giữ Sage throughput.

**Diagnostic trước code model:** Mở rộng `tools/shape14/sage_probe.py` để stream
strict comparison theo query chunks và lưu tối đa một số failure coordinates
`[batch,head,query,channel]`, min/max query, histogram theo prefix cutoff, cùng
`failed_outside_prefix` cho các cutoff định trước. Chạy exact
`B1/H16/S100000/Dh64`, model QKV distribution, model/input seed `1234`, Sage
per-thread INT8-QK + PV-FP16/FP32-accum. Đây là isolated diagnostic; không được
gọi là model accuracy hoặc speedup.

**Gate chọn V19:** (1) nếu có violation ngoài prefix đủ nhỏ (ban đầu cap
`P<=4096`, tức `4.096%` sequence), reject prefix correction trước model code;
(2) nếu locality pass, tạo version riêng V19 kế thừa V18/V16, chạy Sage full
attention rồi exact Flash cho Q prefix và replace đúng rows; non-target/
dependency-missing branch fallback V16; (3) eager official-config B=1 strict
gate với query-blocked baseline, fail một element là dừng; (4) chỉ khi eager
PASS mới sửa custom-op integration bằng `torch.library.opcheck` và
eager/compiled equivalence; (5) batch-limit 2 rồi full 32 strict trước mọi
model performance benchmark; (6) benchmark phải tính cả Sage quant/smoothing,
prefix Flash và copy, so alternating V16/V19 cùng GPU/seed/TF32/warmup/repeats.
Không sửa `main.py` trước khi tất cả gates pass.

## 20. S8.7 — V15.1 direct-layout QKV cross-shape ablation

**Trạng thái:** Completed. Accuracy #1–#12 PASS; chỉ official #6 giữ win có
device-time corroboration. Không promote force-all candidate hay sửa `main.py`.

**Giả thuyết:** V15 giảm official #13 end-to-end `0.98–2.20%` bằng cách thay
packed `[B,S,3D]` interleaved views bằng Triton QKV GEMM ghi thẳng contiguous
`[3,B,H,S,Dh]`. Cùng layout change có thể giúp các official shape khác, nhưng
kernel hiện được autotune từ workload `M=65536,K=128,N=384`; ở M nhỏ, D=32
hoặc D=1024, custom GEMM/extra output traffic có thể chậm hơn compiler/cuBLAS.
Không suy rộng #13 win bằng lý thuyết; đo whole model từng shape.

**Phạm vi:** Tạo `v15_1_DirectQKVAll.py` kế thừa V15, chỉ force
`_use_direct_qkv_layout=True` cho causal configs dưới large-sequence cutoff
`S<8192`. Như vậy exact #14 và memory schedule không đổi; #1–#13 dùng cùng
direct-layout operator để làm ablation. Không sửa V15, V16 hay `main.py`.
Training/non-FP32/non-causal vẫn giữ inherited fallback semantics.

**Gate:** (1) syntax/import/alias, `state_dict`, CPU causal output contract,
training/non-FP32/non-causal fallback và flag coverage D=32/128/1024; (2) RTX
5090 strict accuracy năm trial trên official #1–#12 trước timing (#13 đã có V15
evidence nhưng vẫn là control sanity); (3) chỉ shape PASS mới được paired
max-autotune benchmark V15↔V15.1 bằng cùng seed/TF32/warmup/repeats/rounds;
(4) đảo implementation order cho mọi effect gần noise, ghi cả host median/raw
device evidence; (5) chỉ mở rộng main dispatch cho shape giữ dấu và đủ lớn hơn
variance. Không promote từ isolated QKV timing hay kernel-count reduction.

**Implementation/local gates:** Thêm `v15_1_DirectQKVAll.py` và aliases
`v15.1*`. Candidate chỉ đổi construction-time flag; `state_dict` không đổi.
Syntax/import/alias, flag coverage #1–#14, CPU causal D=32/128/1024,
non-causal/training fallback và Dynamo-eager smoke đều PASS. #1–#13 bật direct
layout; `S=100000` của #14 vẫn tắt, nên không chạm large-sequence path.

**Accuracy gate:** Vast.ai RTX 5090, PyTorch `2.11.0+cu128`, CUDA 12.8,
FP32 public/FP16 internal, TF32 và `compile:max-autotune`. Official #1–#12,
năm trial/shape đều PASS strict: tổng `0/896,942,080` failed elements, max abs
toàn sweep `0.00179085`. Artifact accuracy:
`runs/benchmarks/matrix_v15_1_DirectQKVAll_float32_20260830T142400Z.json`.
Timing `1/1/1` trong artifact này chỉ là orchestration và bị loại.

**Paired sweep:** Mỗi shape chạy process riêng với accuracy ba trial, warmup
`20`, repeats `100`, rounds `3`, profile `3/3`, cùng seed/TF32/compile mode;
sau đó đảo thứ tự V15↔V15.1. Bảng dưới ghi `V15.1 / V15 - 1`; số âm là
candidate nhanh hơn. `E2E A/B` và `GPU A/B` lần lượt là hai implementation
orders.

| Shape | B/S/D/H/FFN/L | E2E A / B | Raw GPU A / B | Quyết định |
|---:|---|---:|---:|---|
| #1 | 64/128/128/4/128/4 | `-1.27% / +0.64%` | `+2.83% / +6.61%` | Reject; order flip và device regress |
| #2 | 1/128/128/4/128/4 | `-0.60% / -0.65%` | `+26.82% / +6.94%` | Retest dài rồi reject |
| #3 | 4/128/128/4/128/4 | `+0.67% / 0.00%` | `+15.88% / +15.59%` | Reject |
| #4 | 16/128/128/4/128/4 | `-1.88% / -1.27%` | `-1.05% / +2.60%` | Retest dài rồi reject |
| #5 | 64/128/128/4/128/4 | `+1.84% / +2.68%` | `+1.82% / +2.71%` | Reject |
| #6 | 10000/128/128/4/128/4 | **`-2.41% / -4.44%`** | **`-3.03% / -2.96%`** | **Winner** |
| #7 | 1/128/1024/16/1024/2 | `0.00% / +0.68%` | `+14.89% / +16.02%` | Reject |
| #8 | 64/128/1024/4/1024/4 | `+2.76% / +3.31%` | `+2.18% / +3.22%` | Reject |
| #9 | 64/128/128/8/128/4 | `-0.03% / -0.04%` | `+1.58% / -0.09%` | Neutral; reject |
| #10 | 64/128/128/2/128/4 | `-0.02% / +0.62%` | `+5.11% / +8.76%` | Reject |
| #11 | 64/128/128/1/128/4 | `+4.34% / +3.32%` | `+3.98% / +2.99%` | Reject |
| #12 | 16/32/128/4/128/4 | `-1.27% / +2.60%` | `+3.75% / -0.47%` | Reject |

Shape #6 medians là `25.9640 → 25.3395 ms` ở V15→V15.1 và, khi đảo order,
`26.0337 → 24.8768 ms`; raw device tương ứng `25.3020 → 24.5348 ms` và
`25.2856 → 24.5382 ms`. Geometric mean của hai ratios là **-3.43% E2E** và
**-2.99% raw GPU**. Candidate thêm bốn direct-QKV kernels nhưng làm Flash time
giảm đủ lớn trên `B*S=1,280,000`; đây là layout-consumer win, không phải chỉ
host timing.

**Retest dài #2/#4:** Hai shape ambiguity được chạy lại cả hai orders với
accuracy năm trial, warmup `100`, repeats `1000`, rounds `7`, profile `5/10`.
#2 E2E là `-0.65%/+1.32%` nhưng raw GPU chậm `+16.10%/+17.62%`; #4 E2E là
`0.00%/+1.25%` và raw GPU chậm `+6.17%/+5.52%`. Vì vậy các apparent wins ở
sweep ngắn là CUDA-graph/host-floor noise, không phải optimization.

**Commands:** Accuracy dùng `tools/matrix_runner.py --impl v15.1 --shape-ids
1,2,3,4,5,6,7,8,9,10,11,12 --accuracy-trials 5 --compile-user --compile-mode
max-autotune`; paired dùng `tools/profile_models.py --impl v15 v15.1 --shape-id ID
--accuracy-trials 3 --warmup 20 --repeats 100 --benchmark-rounds 3
--profile-warmup 3 --profile-iterations 3 --compile-user --compile-mode
max-autotune`, rồi đảo `--impl`. Raw artifacts:
`runs/profiles/profile_shape{01..12}_20260830T143*.json` và
`runs/profiles/profile_shape{01..12}_20260830T144*.json`; retest dài nằm ở
`profile_shape02_20260830T145045Z/145146Z.json` và
`profile_shape04_20260830T145245Z/145345Z.json`.

**Kết luận:** Force direct-layout global là sai: 11/12 shape không thắng đáng
tin cậy. #6 là shape mới duy nhất nên cân nhắc dispatch, ngoài exact #13 đã có.
Không sửa V15/V16/main trong experiment này. Bước promote riêng phải dùng một
workload rule có nghĩa (large `B*S`, D=FFN=128), rerun robustness và xác nhận
aggregate score thay vì thêm silent exact-test tuple.

## 21. S8.8 — V16.1 control giữ #14, bỏ direct-QKV #13

**Trạng thái:** Superseded by source-clean refactor S8.9; kết quả dưới đây là
historical evidence của inherited-control đầu tiên. Không đổi `main.py`.

**Mục tiêu:** Tạo control trả lời chính xác “V16 nếu không có optimization test
#13” mà không làm mất compiled B=1 executor của #14. V14.1 bỏ direct-QKV nhưng
cũng chưa có V16 #14 executor, nên không phải one-variable control của current
main.

**Giả thuyết:** Vì V16 kế thừa V15 và exact-#13 specialization chỉ phụ thuộc
construction-time `_use_direct_qkv_layout`, một wrapper kế thừa V16 rồi force
flag `False` sẽ cho #13 arithmetic/output đúng V14.1 trong khi #14 vẫn chạy
V16 compiled executor. Không cần duplicate forward hay sửa parameter/cache.

**Phạm vi/gate:** Tạo `v16_1_NoDirectQKV13.py`, không sửa V16/`main.py`.
Kiểm tra syntax/import/alias, flag #13/#14, `state_dict`, local exact-#13 output
equivalence với V14.1 sau cùng weights, training/non-FP32/non-causal fallback,
và compile smoke. Sau đó official #13 strict năm trial trên RTX 5090 trước một
paired V14.1↔V16.1 check; candidate phải có cùng graph/timing trong noise. #14
được xác nhận bằng branch/executor identity và existing V16 evidence; không rerun
full 3.2768B accuracy chỉ để kiểm tra một flag vốn đã `False` ở `S=100000`.

**Implementation/local gates:** `v16_1_NoDirectQKV13.py` kế thừa V16 và chỉ
force `_use_direct_qkv_layout=False` sau `super().__init__`. Alias `v16.1*`
được thêm vào matrix/profiler resolver. Syntax/import/alias, exact #13 flag,
#14 flag/executor API, identical `state_dict`, training và Dynamo-eager PASS.
Sau cùng weights, diagnostic #13-config output khớp V14.1 bitwise.

**GPU accuracy:** Vast.ai RTX 5090, PyTorch `2.11.0+cu128`, CUDA 12.8, TF32,
`compile:max-autotune`. Official #13 năm trial PASS strict
`0/41,943,040`, max abs `0.00147235`. Artifact:
`runs/benchmarks/matrix_v16_1_NoDirectQKV13_float32_20260830T150410Z.json`;
timing `1/1/1` trong matrix bị loại.

**Paired control:** V14.1↔V16.1 chạy cả hai orders, accuracy ba trial, warmup
`20`, repeats `100`, rounds `3`, profile `3/3`:

| Order | V14.1 median | V16.1 median | V16.1 Δ | V14.1 raw GPU | V16.1 raw GPU | Raw Δ |
|---|---:|---:|---:|---:|---:|---:|
| V14.1 → V16.1 | 1.1970 ms | 1.2061 ms | +0.76% | 1.0713 ms | 1.0752 ms | +0.36% |
| V16.1 → V14.1 | 1.2105 ms | 1.2124 ms | +0.16% | 1.0792 ms | 1.0983 ms | +1.78% |

Cả hai implementation sinh cùng `29` kernels, hai memory events, `21` Triton
GPU events, một compiled region/CUDA Graph launch và cùng operator sequence.
Chênh latency nhỏ là control drift, không phải performance claim. Artifacts:
`runs/profiles/profile_shape13_20260830T150441Z.json` và
`profile_shape13_20260830T150530Z.json`.

**Kết luận lịch sử:** Bản đầu chứng minh composition đúng về runtime, nhưng còn
dead exact-#13 predicate trong MRO. S8.9 thay source này bằng V14.1 + executor
trực tiếp; không promote và không thay current main.

## 22. S8.9 — Source-clean V16.1 không kế thừa V15

**Trạng thái:** Completed; source/MRO audit và CUDA gates PASS.

**Vấn đề:** V16.1 S8.8 đã tắt direct-QKV ở runtime, nhưng vẫn kế thừa
V16→V15. Vì vậy V15 exact official-#13 tuple vẫn được evaluate trong parent
constructor rồi mới bị overwrite `False`; source/MRO cũng còn direct-QKV code.
Đây là dead dispatch, không ảnh hưởng graph, nhưng không đạt mục tiêu audit
“không có test-based branch” ở cấp source.

**Giả thuyết/refactor:** Cho `v16_1_NoDirectQKV13.py` kế thừa trực tiếp V14.1
và chứa riêng executor/cache methods đã validate của V16. Khi đó #1–#13 dùng
V14.1/V11 arithmetic không qua V15 import; #14 vẫn dùng cùng cutoff, outer
compiler-disabled batch loop và reusable compiled B=1 executor. Không sửa
V16/main để refactor control không tạo regression risk cho stable path.

**Gate:** (1) `rg`/MRO xác nhận V16.1 không import/inherit V15, không chứa exact
#13 tuple và direct-QKV symbol; (2) state-dict và #13 output khớp V14.1 bitwise,
mọi official config có force-all V11 FFN flag và không có QKV flag; (3) #14
executor/cache lifecycle, invalidation, B=1 validation và Dynamo-eager smoke
khớp V16; (4) rerun official #13 strict năm trial và paired graph control; (5)
GPU targeted `S>=8192` executor equivalence với V16 trên manageable diagnostic.
Không đổi `main.py`.

**Implementation/source audit:** V16.1 giờ import trực tiếp
`v14_1_BatchChunked.UserOptimizedTransformer`; không import V15/V16 và không
có `_use_direct_qkv_layout`. MRO thực tế là
`V16.1 → V14.1 → V11 → V8 → V4.3 → mixed → baseline`. `rg` trên file không
thấy exact config comparisons cho B/S/D/H/L/FFN. Trên cả 14 official configs,
direct-QKV attribute không tồn tại và V11 fused-FFN flag luôn `True`; predicate
large-token lịch sử của V8 bị V11 override universal như trước.

Executor methods được copy giữ nguyên từ validated V16 nhưng parent là V14.1.
Local exact-#13 config khớp V14.1 bitwise. Forced-cutoff B2/S8 diagnostic có
prefix mask khớp V14.1 bitwise; executor backend-eager build/reuse và invalidation
sau `load_state_dict`, `_apply`/`.to()` và `train()` đều PASS.

**CUDA #13 revalidation:** RTX 5090/PyTorch `2.11.0+cu128`, official #13 năm
trial PASS strict `0/41,943,040`, max abs `0.00147235`. Paired V14.1↔clean
V16.1, warmup/repeats/rounds `20/100/3`, chạy cả hai orders:

| Order | V14.1 median | Clean V16.1 median | Delta | Raw GPU delta | Graph |
|---|---:|---:|---:|---:|---|
| V14.1 → V16.1 | 1.1977 ms | 1.2089 ms | +0.93% | +0.08% | 29 kernels / 21 Triton events |
| V16.1 → V14.1 | 1.2219 ms | 1.2099 ms | -0.98% | -5.69% | 29 kernels / 21 Triton events |

Host delta đổi dấu và graph/operator sequence giống hệt; raw reverse bị control
drift nên không dùng làm win. Artifacts mới:
`runs/benchmarks/matrix_v16_1_NoDirectQKV13_float32_20260830T151525Z.json`,
`runs/profiles/profile_shape13_20260830T151536Z.json` và
`profile_shape13_20260830T151623Z.json`.

**CUDA #14 executor canary:** Exact config #14, compiled inner executor,
`batch-limit=1`, query/compare chunks `256/2048`, seed 1234 PASS strict
`0/102,400,000`, max abs `0.000719786`, mean abs `6.56403e-05`, elapsed
`16.071 s`, peak `19.967 GiB`. Command:

```bash
CUDA_VISIBLE_DEVICES=0 /venv/main/bin/python -m tools.shape14.accuracy \
  --device cuda:0 --impl v16.1 --batch-limit 1 --query-chunk 256 \
  --compare-token-chunk 2048 --seed 1234
```

**Kết luận:** V16.1 hiện source-clean ngoài intentional large-sequence/#14
dispatch: không exact official tuple, không V15 dependency, không dead
direct-QKV flag. #13 giữ V14.1 graph và #14 giữ V16-equivalent compiled sample
schedule. Ở thời điểm S8.9, `main.py` vẫn V16; S8.10 bên dưới ghi nhận lần
promote V16.1 sau đó.

## 23. S8.10 — Promote V16.1 làm source-clean main

**Trạng thái:** Historical promotion checkpoint theo D-038; full clean-artifact
GPU gate về sau đã được đóng trong S8.13/D-043.

**Mục tiêu và giả thuyết:** Dùng V16.1 làm stable submission artifact để #1–#13
không chứa exact official-#13 tuple hoặc V15/direct-QKV dependency, đồng thời giữ
memory-bounded compiled B=1 schedule của V16 cho #14. Đây là architecture/source
promotion, không phải giả thuyết V16.1 nhanh hơn V16.

**Trade-off đã biết:** V16 direct-QKV #13 từng thắng V14.1 `0.98–2.20%`
end-to-end và `1.17–1.76%` raw GPU. V16.1 khớp graph V14.1, vì vậy promotion
chủ đích bỏ gain này để loại test-specific source branch. Không gán lại V16
full-#14 numbers cho V16.1: schedule source được copy, nhưng evidence V16.1 hiện
chỉ có exact-config compiled B=1 canary.

**Evidence dùng cho quyết định:** Official #13 V16.1 PASS strict năm trial
`0/41,943,040`, max abs `0.00147235`; paired V14.1/V16.1 cùng `29` kernels và
`21` Triton events, host delta đổi dấu `+0.93%/-0.98%`. Exact-config #14 B=1
canary PASS `0/102,400,000`, max abs `0.000719786`, peak `19.967 GiB`. Source/MRO
audit không thấy V15/V16 import, direct-QKV symbol hoặc exact official tuple.

**Thay đổi:** `main.py` import `UserOptimizedTransformer` từ
`v16_1_NoDirectQKV13.py`; stable aliases `main`/`best` vẫn resolve qua
`main.py`. V16 được giữ nguyên làm rollback có direct-QKV #13. Không đổi
baseline, public API, state dict, cutoff/chunk size, tolerance, comparator hay
workload.

**Local validation:** PyTorch `2.12.1`, CPU; `torch.cuda.is_available()` là
`False`, nên không tạo GPU performance claim. Full repository `py_compile`
theo `AGENTS.md`, `tools/matrix_runner.py --list-shapes`,
`tools/profile_models.py --list-shapes` và `--help` của ba tool #14 đều PASS. Các
tool `tools/shape14/accuracy.py`, `tools/shape14/optimized_benchmark.py` và
`tools/shape14/profile.py` nay nhận/default alias `main` và resolve đúng V16.1.
Identity/MRO audit xác nhận `main.UserOptimizedTransformer is V16.1` và chain
không chứa V15/V16. CPU official-shape #2 smoke qua `main.py` PASS hai trial,
`0/32,768`, max abs `0.000937343`; matrix-runner alias `main` PASS một trial,
max abs `0.00084424`. CPU timing `0.450–0.506x` là non-target orchestration
diagnostic và bị loại khỏi mọi performance claim.

**Gate tại thời điểm promotion:** cần full V16.1 #14 strict đủ 32 batch và full
#1–#13 main matrix trên GPU mục tiêu idle. S8.13/D-043 đã đóng hai gate bắt buộc
này; alternating V16.1/V16/V14.1 chỉ còn là optional predecessor attribution,
không cần để claim active-main latency.

**Hướng tối ưu kế tiếp theo evidence:** (1) candidate direct-QKV mới trên V16.1
với workload predicate large `B*S`, `D=FFN=128` để khai thác measured #6 win
`3.43%` E2E mà không hard-code #13; (2) shape #14 failure-locality probe rồi
selective exact-prefix correction cho Sage nếu mọi violation nằm trong prefix
nhỏ; (3) exact FlashInfer SM120 shoot-out trước khi viết custom attention.

## 24. S8.11 — V17-Sage cross-shape với exact-prefix correction

**Trạng thái:** Rejected sau full #1–#13 GPU matrix: official #6 và #9 fail
strict accuracy. Không thay `main.py`.

**Evidence và giả thuyết:** Artifact
`runs/profiles/shape14_sage_locality_seed1234_20260830.json` đã hoàn tất phần
diagnostic mà S8.6 còn ghi planned. Trên exact causal attention
`B1/H16/S100000/Dh64`, model-QKV distribution và seed 1234, Sage
INT8-QK per-thread + PV-FP16/FP32-accum nhanh isolated `1.3915x` nhưng fail
`109/102,400,000`. Toàn bộ violation nằm ở query `1..31`; `minimal_exact_prefix`
là `32` và `failed_outside_prefix[32] == 0`. Do causal semantics, 32 query đầu
chỉ phụ thuộc 32 key/value đầu, nên có thể tính chính xác square prefix bằng
PyTorch Flash rồi overwrite đúng 32 output rows; phần suffix vẫn dùng Sage.

**Phạm vi candidate:** Tạo file theo tên owner yêu cầu `v17_sage.py`, gọi trong
report là V17-Sage để không nhầm với historical `v17_CompiledBatch2.py`/V17-B2.
Candidate kế thừa source-clean V16.1, không kế thừa V18/V16/V15. Sage được thử
trên causal FP32-eval config có `S > P` và `head_dim <= 128`; API pin hiện tự
pad head dimension nhỏ lên 64/128. `head_dim > 128`, `S <= P`, training,
non-FP32, non-causal, CPU hoặc thiếu optional dependency fallback nguyên V16.1.
Đây là workload/support-envelope predicate, không phải exact official tuple.

**Accuracy guards:** Recipe Sage giữ `qk_quant_gran="per_thread"`,
`pv_accum_dtype="fp32"`, `smooth_k=True`; exact causal prefix mặc định `P=32`;
attention out-projection dùng FP16 operands nhưng store FP32 accumulator theo
helper V12.1 để mua thêm residual-boundary margin. Custom op được đánh dấu
`torch.Tag.cudagraph_unsafe` vì upstream và V18 evidence đều cho thấy Sage
custom op sai khi CUDA Graph capture; fake output contract là contiguous HND.
`TECHJAM_SAGE_EXACT_PREFIX` cho phép accuracy ablation và
`TECHJAM_SAGE_REQUIRE=1` cấm silent dependency fallback trong official runs.

**Gate:** (1) syntax/import/state-dict và CPU fallback smoke; (2) trên target,
`torch.library.opcheck` cùng eager/custom-op output contract; (3) eager và
compiled no-CUDA-Graph equivalence trên sample nhẹ; (4) full official #1–#13
strict matrix, ghi rõ #8 (`head_dim=256`) và #12 (`S=32`) là safe fallback;
(5) shape #14 B=1 canary, rồi full 32 strict gate; (6) chỉ sau toàn bộ accuracy
PASS mới chạy performance matrix/alternating comparison. Theo yêu cầu owner,
implementation turn này không chạy full benchmark; chỉ local smoke nhẹ và bàn
giao command GPU.

**Implementation/local validation:** `v17_sage.py` kế thừa trực tiếp V16.1;
custom op có contiguous FakeTensor contract và `cudagraph_unsafe` tag. Aliases
`v17.sage`/`v17_sage` đã nối vào matrix/profile và ba tool #14 mà không đổi
alias historical `v17`. Official construction-time support map bật Sage ở
#1–#7, #9–#11, #13–#14; #8 (`Dh=256`) và #12 (`S=32`) fallback. Local PyTorch
2.12.1 CPU: syntax/import, exact state-dict equality, dependency-required guard,
causal square-prefix equivalence và V16.1 fallback equality PASS. Official #2
CPU smoke hai trial PASS `0/32,768`, max abs `0.000937343`; timing một repeat
`0.631x` là non-target orchestration diagnostic và không phải performance
evidence. Host không có CUDA/Sage, nên chưa execute Sage kernel, `opcheck` hoặc
compiled-eager comparison và không tạo speedup claim.

`v17_sage_opcheck.py` là accuracy/integration-only preflight: kiểm tra đúng
SageAttention 2.2.0, chạy `torch.library.opcheck`, compile custom op bằng
`max-autotune-no-cudagraphs`, rồi áp strict comparator giữa eager/compiled.
Script không đo latency và phải PASS trước targeted model canaries.

**GPU matrix 2026-08-31:** RTX 5090, PyTorch `2.11.0+cu128`, SageAttention
`2.2.0`, FP32 public input/output, TF32 bật, seed `1234`, accuracy trials `5`,
warmup/repeats/rounds `20/100/3`, optimized compile `max-autotune`. Artifact
remote: `runs/benchmarks/matrix_v17_sage_float32_20260831T034305Z.{json,csv}`.

| ID | Accuracy | Max abs | Baseline ms | Optimized ms | Speedup |
|---:|---|---:|---:|---:|---:|
| 1 | PASS | 0.00201304 | 2.6558 | 2.1926 | 1.211x |
| 2 | PASS | 0.00177810 | 2.5521 | 2.1171 | 1.205x |
| 3 | PASS | 0.00193930 | 2.6572 | 2.1284 | 1.248x |
| 4 | PASS | 0.00205994 | 2.5783 | 2.1284 | 1.211x |
| 5 | PASS | 0.00227185 | 2.6107 | 2.1669 | 1.205x |
| 6 | **FAIL** | 0.00250164 | — | — | — |
| 7 | PASS | 0.00246114 | 2.5814 | 2.1648 | 1.192x |
| 8 | PASS, V16.1 fallback | 0.00134873 | 6.7651 | 2.9503 | 2.293x |
| 9 | **FAIL** | 0.00255397 | — | — | — |
| 10 | PASS | 0.00234208 | 2.5973 | 1.8344 | 1.416x |
| 11 | PASS | 0.00187761 | 2.6936 | 2.1472 | 1.254x |
| 12 | PASS, V16.1 fallback | 0.00127272 | 2.6152 | 0.1618 | 16.161x |
| 13 | PASS | 0.00207222 | 41.9384 | 2.4573 | 17.067x |

Kết quả xác nhận exact-prefix correction không đủ robust cross-shape và Sage
overhead áp đảo ở `S=128`. #13 có upside lớn nhưng không cứu được official
candidate vì correctness là gate toàn cục. V17-Sage giữ làm negative ablation;
V16.1 tiếp tục là main.

## 25. S8.12 — V18-Sage direct automatic SM120 performance probe

**Trạng thái:** Planned/implementing theo yêu cầu owner; performance-only
diagnostic, không phải promotion candidate và không đổi `main.py`.

**Giả thuyết:** V17-Sage trộn ba chi phí ngoài raw backend: PV-FP16 recipe,
exact Flash prefix/copy và FP32 attention out-projection. Để đo trần throughput
của Sage trên RTX 5090, V18-Sage kế thừa trực tiếp source-clean V16.1 và chỉ
thay SDPA bằng `sageattention.sageattn` automatic. Ở SM120, source pin dispatch
automatic sang INT8-QK per-warp + FP8-PV `fp32+fp16`; không exact-prefix và
không FP32 out-projection correction.

**Phạm vi:** Causal FP32 eval/CUDA với original `head_dim<=128` dùng Sage trực
tiếp. Official #8 có `head_dim=256`, vượt support envelope của Sage 2.2 nên
fallback V16.1; mọi training/CPU/non-FP32/non-causal branch cũng fallback.
Official #12 `S=32` vẫn chạy Sage để đo overhead thật. Dependency guard/version
pin và CUDA-Graph-unsafe wrapper vẫn giữ để không đo silent fallback hoặc replay
sai. Public API, parameter/state dict và V16.1 large-sequence schedule giữ nguyên.

**Measurement policy:** Vì owner chủ đích chưa gate numerical accuracy, timing
trên case FAIL chỉ là invalid diagnostic. Thêm opt-in
`tools/matrix_runner.py --benchmark-on-failure` để forward flag đã có sẵn của official
harness; comparator/tolerance/baseline không đổi và status vẫn phải ghi
`ACCURACY_FAIL`. Agent chỉ chạy syntax, CPU fallback/integration smoke và một
sample nhẹ; owner chạy full GPU benchmark.

## 26. S8.13 — Standalone V16.1 cleanup và archive

**Trạng thái:** Validated and benchmarked; local equivalence và fresh GPU
official matrix đều PASS.

**Giả thuyết/phạm vi:** Flatten đúng promoted V16.1 vào một file standalone để
artifact submission không còn phụ thuộc chuỗi V14.1→V11→V8→V4.3→mixed→baseline.
Đây là packaging/source-topology change: giữ nguyên parameter names, FP16 cache,
Flash-first SDPA, FP32-pre-GELU Triton arithmetic, fallback, cutoff `8192`, batch
chunk `1` và compiled executor policy. Không tạo performance claim mới.

**Implementation:** `v16_1_clean.py` chỉ import PyTorch và optional Triton.
`main.py` gắn class clean vào harness. Matrix/profile aliases active được rút
gọn còn `main`/`best`/`v16.1`; shape-#14 tools cũng chỉ dùng class clean. Toàn
bộ 35 file `v*.py` lịch sử khác chuyển vào `archive/versions/`.

**Local gates, PyTorch CPU:** `py_compile` PASS. Strict `state_dict` giữa
composed V16.1 cũ và clean không có missing/unexpected key. Causal/non-causal ×
mask/no-mask eval khớp bit-for-bit; training FP32/BF16 fallback khớp bit-for-bit.
Forced large-sequence cutoff với `torch.compile(backend="eager")` cũng khớp
bit-for-bit; compiled executor được reuse rồi invalidate đúng sau
`load_state_dict()` và `.to()`.

**GPU gate cần đóng sau cleanup:** chạy active syntax/list-shapes/main smoke,
CUDA official #1–#13 và full memory-bounded #14 trên GPU mục tiêu. Trước khi có
run ngày 2026-08-31, local CPU equivalence không được dùng để gán lại
latency/speedup predecessor cho file clean.

**Post-archive active smoke:** Active `py_compile`, hai lệnh `--list-shapes` và
`--help` của ba tool #14 đều PASS. `main.py` trên official shape #2 CPU PASS
hai trial (`0/32,768`, `max_abs=0.000937343`); matrix alias `v16.1` PASS một
trial (`0/16,384`, `max_abs=0.00084424`). CPU timing bị loại. Copy duy nhất
`v16_1_clean.py` sang temp directory và import khi repo path bị loại khỏi
`sys.path` cũng PASS, xác nhận standalone import không dựa vào file project.

**Fresh final GPU validation 2026-08-31:** Commit
`4f77a04fd51c3a2ad8b9d8986657915ae3ca94d6` được chạy trên Vast.ai RTX 5090
`sm120`, driver `580.159.03`, Ubuntu 24.04.4, Python `3.12.14`, PyTorch
`2.11.0+cu128`, CUDA wheel `12.8`, cuDNN `9.19.0`, Triton `3.6.0`. Public dtype
là FP32, TF32 bật cho cả baseline/optimized, seed `1234`; baseline eager còn
optimized dùng `max-autotune`.

Command #1–#13:

```bash
CUDA_VISIBLE_DEVICES=0 /venv/main/bin/python -m tools.matrix_runner \
  --impl main --shape-ids 1,2,3,4,5,6,7,8,9,10,11,12,13 \
  --device cuda:0 --dtype float32 --accuracy-trials 5 \
  --warmup 20 --repeats 100 --benchmark-rounds 3 \
  --compile-user --compile-mode max-autotune --timeout 1800
```

Mọi row strict PASS, tổng failed `0`, worst max abs `0.00179085`; geomean
speedup `7.904x`.

| ID | Max abs | Baseline ms | Optimized ms | Speedup |
|---:|---:|---:|---:|---:|
| 1 | 0.00127272 | 0.8146 | 0.1389 | 5.863x |
| 2 | 0.00114284 | 0.7136 | 0.0566 | 12.606x |
| 3 | 0.00123456 | 0.7764 | 0.0607 | 12.789x |
| 4 | 0.00134718 | 0.7214 | 0.0771 | 9.354x |
| 5 | 0.00147235 | 1.3092 | 0.2208 | 5.930x |
| 6 | 0.00160612 | 179.0890 | 26.5491 | 6.746x |
| 7 | 0.00179085 | 0.7208 | 0.0732 | 9.853x |
| 8 | 0.00134873 | 7.1229 | 2.8616 | 2.489x |
| 9 | 0.00126606 | 0.6540 | 0.1574 | 4.155x |
| 10 | 0.00134726 | 0.7254 | 0.1451 | 5.001x |
| 11 | 0.00140083 | 1.8169 | 0.1996 | 9.105x |
| 12 | 0.00134718 | 0.7213 | 0.0854 | 8.442x |
| 13 | 0.00147235 | 42.2520 | 1.2455 | 33.925x |

Full #14 dùng `tools/shape14/accuracy.py --impl main --batch-limit 32
--query-chunk 256 --compare-token-chunk 2048 --seed 1234 --compile-mode
max-autotune`: strict PASS `0/3,276,800,000`, max abs `0.000944197`, mean abs
`6.56367e-05`, elapsed `348.534 s`, peak accuracy allocation `19.967 GiB`.

Optimized-only #14 với warmup/repeats `1/5` cho samples
`7135.0098/7171.5771/7213.5254/7239.6484/7260.9688 ms`; median
`7213.5254 ms`, p90 `7252.4406 ms`, throughput `443,611.11 token/s`, peak
`24.487 GiB`. Baseline/speedup giữ N/A vì explicit score khoảng `18.6 TiB`.

Checked-in evidence: `results/final/`.

## 27. S8.14 — V19 CUDA FP16 accumulation với checkpoint FP32 theo K

**Trạng thái:** GPU sweep complete; correctness PASS nhưng performance regress,
không promote. `main.py` vẫn V16.1.

**Giả thuyết:** FFN-in/GELU của V16.1 hiện dùng custom Triton GEMM với
accumulator FP32. Full FP16 accumulation từng fail accuracy ở V5.1 và không cho
speedup ổn định, nhưng lỗi tích lũy có thể được giới hạn nếu mỗi tensor-core
tile chỉ accumulate FP16 trong một đoạn K ngắn rồi promote partial sum vào một
accumulator FP32. Candidate đầu tiên chốt sau mỗi `K=32`: hai MMA K=16 dùng
accumulator FP16, partial tile được store/promote, cộng thủ công vào FP32, rồi
reset accumulator FP16 cho đoạn tiếp theo. Bias và exact erf-GELU vẫn chạy từ
tổng FP32; output GELU vẫn round FP16 cho FFN-out như V16.1.

**Phạm vi:** Tạo version riêng `candidates/v19/cuda_fp16_checkpoint.py` kế thừa standalone
V16.1 và chỉ thay fused `FFN-in -> exact GELU`. CUDA extension dùng WMMA 16x16x16
thật, không bật flag process-global `allow_fp16_accumulation`. Default là
checkpoint `K=32`; các control `K=16/64/128` và `fp32` dùng cùng launch/layout/
epilogue để owner sweep trên GPU sau. QKV, SDPA, attention out, FFN-out,
LayerNorm/residual, cache/state dict, large-sequence executor và mọi fallback
khác giữ nguyên V16.1. `main.py` tiếp tục trỏ V16.1 cho tới khi V19 pass strict
matrix và paired benchmark.

**Gate:** (1) syntax/import và strict state-dict equality; (2) local CPU
portable checkpoint simulation, causal/non-causal, mask/no-mask, training và
non-FP32 fallback; (3) trên target, bắt buộc build/chạy CUDA extension và so
kernel `K=16/32/64/128/fp32` trên official canaries #7/#10/#2 trước; (4) chỉ
variant strict PASS mới chạy #1–#13 rồi #14 B=1 -> B=2 -> B=32; (5) benchmark
paired V16.1/V19 bằng cùng seed/TF32/max-autotune và cả hai orders. Local CPU
timing và mọi CUDA timing khi GPU đang có job đều bị loại khỏi performance
claim.

**Implementation:** CUDA kernel dùng block `64x32`, tám warps và WMMA
`16x16x16`. Activation tile và transposed-weight view được stage vào shared
memory; mỗi warp giữ một output tile. Ở mode mặc định `K=32`, hai MMA cập nhật
accumulator FP16, partial tile được store vào shared memory, mỗi lane promote
tám phần tử sang register FP32 rồi accumulator FP16 được reset. Các mode
`16/64/128` thay đúng số MMA giữa hai checkpoint; `fp32` là CUDA control dùng
cùng block/layout/epilogue nhưng WMMA accumulator FP32. Extension được build
trước `torch.compile` khi model chuyển sang CUDA. Build failure mặc định là
hard error để không silent benchmark V16.1; chỉ env
`TECHJAM_V19_ALLOW_CUDA_FALLBACK=1` cho phép diagnostic fallback có warning.

**Local result:** macOS/CPU, PyTorch `2.12.1`; không có CUDA nên phần này chỉ
validate Python graph, state/fallback và portable checkpoint simulation.
`py_compile`, import, custom-op `torch.library.opcheck` (schema/autograd/fake/
AOT dynamic) và `torch.compile(backend="eager")` đều PASS. State dict khớp
V16.1 strict; unsupported-M, training và BF16 fallbacks khớp V16.1 bitwise.
Causal/non-causal x mask/no-mask diagnostic `B2/S16/D32/H4/FFN32/L2` đều PASS
strict, worst max abs `0.000994802`. Official-shape-#2 CPU một-trial smoke PASS
cho cả năm modes:

| Mode | Max abs | Failed |
|---|---:|---:|
| K=16 | `0.000965476` | `0/16,384` |
| K=32 | `0.00108075` | `0/16,384` |
| K=64 | `0.00101018` | `0/16,384` |
| K=128 | `0.00104527` | `0/16,384` |
| FP32 control | `0.00084424` | `0/16,384` |

CPU latency bị loại. Portable path chỉ round partial GEMM output theo group;
nó không mô phỏng chính xác vi kiến trúc WMMA FP16 accumulator, nên các PASS
trên không thay CUDA accuracy gate.

**GPU result 2026-09-01:** Vast.ai RTX 5090 SM120, driver `595.71.05`, Python
`3.12.14`, PyTorch `2.11.0+cu128`, CUDA wheel `12.8`, cuDNN `9.19.0`, Triton
`3.6.0`; seed `1234`, FP32 public, TF32 bật. NVCC 12.8 bắt được một type mismatch
ở `wmma::fill_fragment` cho FP16 accumulator; literal zero đã đổi thành
`__float2half(0.0f)`, sau đó extension build/chạy thật, không fallback.

Mọi K=16/32/64/128/fp32 strict PASS năm trials trên #7/#10/#2. Shape #6
max-autotune `10/30/3`: V16.1 controls `25.1593/25.2380 ms`; K16/K32/K64/K128
lần lượt `29.7888/29.8112/29.7675/29.8196 ms`; FP32 control `29.5313 ms`.
K64 là FP16 mode nhanh nhất nhưng regress khoảng `18.13%` so V16.1.

K64 full #1–#13, command `tools/matrix_runner.py --impl v19 --shape-ids 1,...,13
--accuracy-trials 5 --warmup 20 --repeats 100 --benchmark-rounds 3
--compile-user --compile-mode max-autotune`: strict PASS 13/13, worst max abs
`0.00181192`, geomean speedup `10.3079x`. V16.1 start control cùng host là
`11.8030x`; direct optimized-latency geomean của V19 regress `13.73%`.

Full #14 K64 PASS `0/3,276,800,000`, max abs `0.000997305`; two-order
optimized-only medians `7251.4170/7310.5811 ms`. V19 bị reject cho promotion.
Raw evidence và command-level report:
`results/experiments/v19-tuning-20260901/`.

## 28. S8.15 — V19.1 parallel batch partitions

**Trạng thái:** GPU tuning complete. V19.1.0 P4 là measured winner cho shape
#14; không tự động promote `main.py` vì owner chỉ yêu cầu tune/report.

**Evidence và giả thuyết:** V16.1 shape #14 giữ loop batch ngoài eager và chạy
32 sample B=1 tuần tự. V17 từng đổi executor thành B=2 nhưng chỉ giảm khoảng
`0.515%`; cách đó tăng batch bên trong cùng graph, không thử enqueue các sample
độc lập lên nhiều CUDA streams. Vì batch samples không tương tác, chia B=32
thành 2/4/8/... partition liên tục và chạy mỗi partition trên một stream có thể
overlap attention/GEMM hoặc che launch gaps. Đổi lại, live intermediates tăng
gần theo số stream và có thể OOM; đây là memory/performance experiment, không
được giả định nhanh hơn từ concurrency lý thuyết.

**V19.1.0:** Kế thừa trực tiếp standalone V16.1. Chỉ thay outer large-sequence
batch scheduler bằng multi-stream partitions; arithmetic/kernel bên trong mỗi
sample giữ nguyên V16.1.

**V19.1.1:** Kế thừa V19 và dùng đúng scheduler V19.1.0. Vì vậy nó kết hợp hai
thay đổi đã version hóa: CUDA checkpointed-FP16 FFN-in/GELU của V19 và batch
partitions song song. So V19.1.1 với V19 để đo riêng scheduler; so V19.1.0 với
V16.1 làm control không có kernel V19.

**Thiết kế:** `TECHJAM_V19_PARALLEL_PARTS=1|2|4|8|16|32`, mặc định 2. Batch
được chia thành các range liên tục cân bằng; mỗi stream xử lý tuần tự các sample
B=1 trong range của nó, còn các stream enqueue song song. Parts=1 gọi nguyên
parent path. Multi-stream bắt buộc inner executor dùng
`max-autotune-no-cudagraphs`: một CUDA Graph/static buffer dùng chung giữa các
stream có nguy cơ race hoặc replay sai. Streams là runtime cache không thuộc
`state_dict`, được rebuild sau load/move/train-eval invalidation. CPU/non-CUDA,
training, non-FP32, short sequence và B=1 giữ parent behavior.

**Gate:** (1) planner cover mỗi batch index đúng một lần cho B lẻ/chẵn và mọi
parts; (2) syntax/import/aliases, state dict và parent fallback bitwise; (3)
CUDA strict #14 B=2 canary cho parts=2, rồi B=4/8/16/32 theo memory gate; OOM là
kết quả hợp lệ và dừng tăng parts; (4) full #14 strict B=32 trước timing; (5)
alternating optimized-only V16.1/V19.1.0 và V19/V19.1.1, cùng
seed/TF32/no-CUDA-Graph/warmup/repeats; (6) báo peak memory cùng latency. Không
chạy parts 4/8 chỉ vì parts 2 PASS nếu headroom không đủ, và không dùng local
CPU timing làm performance evidence.

**Implementation và local result:** Scheduler chung nằm trong
`candidates/v19/parallel_batch_common.py`; hai entrypoint là
`candidates/v19/parallel_batch_v161.py` và `candidates/v19/parallel_batch_v19.py`. Aliases
`v19.1.0`/`v19_1_0` và `v19.1.1`/`v19_1_1` đã nối vào matrix và ba tool #14.
`tools/shape14/accuracy.py` gọi candidate theo group `min(parts, batch-limit)` khi
parts>1 và B>1, thay vì bypass outer scheduler bằng B=1 sample helper. Grouping
exercise đủ worker streams nhưng không giữ full B32 output trong lúc dựng
memory-bounded reference.

Local macOS/CPU PyTorch `2.12.1`: `py_compile` PASS; planner cover chính xác
B=1/2/3/7/32/33 cho mọi parts, không range rỗng và độ lệch width tối đa 1.
Mọi parts hợp lệ parse/configure đúng, parts>1 ép
`max-autotune-no-cudagraphs`, invalid parts bị reject. State-dict keys khớp
đúng parent; forced-large CPU path, mask/no-mask, training và BF16 đều khớp
parent bitwise cho cả hai candidate. Official #2 one-trial smoke PASS:

| Candidate | Parent | Max abs | Failed |
|---|---|---:|---:|
| V19.1.0 | V16.1 | `0.00084424` | `0/16,384` |
| V19.1.1 | V19 K=32 portable | `0.00108075` | `0/16,384` |

CPU timing bị loại: shape #2 có B=1/S=128 nên không đi vào parallel
large-sequence path, và CPU không thể validate CUDA stream overlap.

**GPU result 2026-09-01:** Cùng RTX 5090/driver/PyTorch environment của S8.14.
P2/P4/P8 multi-stream canaries strict PASS; P8 full-output timing regress và
resident memory lên khoảng `29.6 GiB`, nên không thử P16. V19.1.0 sweep hai
orders cho P2 trung bình `6820.9448 ms`, P4 `6810.4595 ms`; chênh `0.15%` nhưng
P4 ổn định hơn và là số nhanh nhất đo được. Post-gate P4 final:

- Full #14 strict PASS `0/3,276,800,000`, max abs `0.000944138`, mean abs
  `6.56367e-05`, accuracy peak `21.147 GiB`.
- Optimized-only warmup/repeats `1/5`: median `6780.3867 ms`, p90
  `6792.4046 ms`, throughput `471,949.48 token/s`, peak `25.676 GiB`.
- Hai P1 control sandwiches cho thấy P4 nhanh hơn V16.1 parent khoảng
  `1.51–1.66%` trên cùng `max-autotune-no-cudagraphs`.

V19.1.1 K64 chọn P2: full #14 PASS `0/3,276,800,000`, max abs `0.000997305`;
two-order medians `7173.3130/7185.5513 ms`. V19.1.0 P4 nhanh hơn V19 K64 P1
khoảng `6.88%` và nhanh hơn V19.1.1 K64/P2 khoảng `5.56%`. `main.py` chưa đổi.
Chi tiết từng run nằm trong
`results/experiments/v19-tuning-20260901/REPORT.md`.

## 29. S8.16 — Full checkpoint timeline trên RTX 5090 driver 595

**Trạng thái:** Complete. Full #1–#13, V16.1 start/end controls,
reverse-order repeats và Baseline/V16.1 #14 đã có curated artifacts.

**Giả thuyết/phạm vi:** Historical report cần đo toàn bộ checkpoint chính trên
cùng host thay vì ghép các số từ nhiều phase/máy. Chạy Baseline, V1, V2,
V3.1 eager, V3.1 compiled, V4.1, V4.2, V4.3, V8, V11 và V16.1 trên đủ official
#1–#13. Correctness strict năm trial là gate trước timing `20/100/3`;
compiled checkpoints dùng `max-autotune`. V16.1 được predeclare làm control
đầu/cuối, drift budget 3%. Theo scope cuối của owner, #14 chỉ chạy Baseline và
V16.1; historical checkpoint #14 sweep đã bị dừng và không report.

**Environment:** Vast.ai RTX 5090 `sm120`, 32,607 MiB, driver `595.71.05`;
AMD Ryzen 5 5600X, 12 logical CPUs, 33,564,246,016 bytes RAM; Python `3.12.14`,
PyTorch `2.11.0+cu128`, CUDA `12.8`, cuDNN `9.19.0`, Triton `3.6.0`. Source
snapshot `4f77a04fd51c3a2ad8b9d8986657915ae3ca94d6`; remote snapshot không có
`.git`, nên manifest dùng declared revision cùng SHA-256 source/dependencies.

**Command chính:** `tools/timeline_runner.py --checkpoints
v16_1,baseline,v1,v2,v3_1_eager,v3_1_compiled,v4_1,v4_2,v4_3,v8,v11,v16_1
--shape-ids 1-13 --accuracy-trials 5 --warmup 20 --repeats 100
--benchmark-rounds 3 --seed 1234 --timeout 1800 --compile-mode max-autotune
--control-drift-threshold 0.03`. Exact executable, environment ID và revision
nằm trong `results/timeline/run_metadata.json`.

| Checkpoint | Accuracy | Forward geomean | Reverse geomean |
|---|---:|---:|---:|
| Baseline | 13/13 PASS | `1.0011x` | — |
| V1 | 13/13 PASS | `1.0763x` | — |
| V2 | 13/13 PASS | `1.4353x` | — |
| V3.1 eager | 13/13 PASS | `2.1006x` | — |
| V3.1 compiled | 0/13 | N/A | — |
| V4.1 | 13/13 PASS | `10.1999x` | `10.1926x` |
| V4.2 | 13/13 PASS | `10.4489x` | `10.4349x` |
| V4.3 | 13/13 PASS | `11.6755x` | `11.6948x` |
| V8 | 13/13 PASS | `11.7854x` | `11.6580x` |
| V11 | 13/13 PASS | `11.7483x` | `11.7439x` |
| V16.1 start | 13/13 PASS | `11.8030x` | `11.7617x` |
| V16.1 end | 13/13 PASS | `11.8383x` | — |

V3.1 compiled fail strict cả 13 shapes với tổng `201,682` failed elements; mọi
timing bị skip đúng gate. V16.1 baseline/optimized geomean drift lần lượt
`0.166%`/`0.458%`; #6/#8/#13 đều dưới 3%, max `1.042%` ở optimized #8.
V4.2 hơn V4.1 ổn định `2.38–2.44%` qua hai orders. V4.3/V8 và V8/V11 đổi dấu
theo order; V11/V16.1 chỉ khác `0.15–0.47%`, nên không gán noise thành gain.

**Shape #14:** Baseline `INFEASIBLE_STATIC`, latency/speedup N/A. V16.1 B1
PASS `0/102,400,000`; streamed B32 PASS `0/3,276,800,000`, max abs
`0.000944197`; native full B32 output contract PASS. Samples optimized-only
`6987.4644/6983.9238/6987.4033/6992.8545/6994.9302 ms`; median
`6987.4644 ms`, p90 `6994.0999 ms`, throughput `457,962.98 token/s`, peak
`24.487 GiB`.

**Reporting decision:** Promote V16.1 start-control `11.803x`, không chọn end
control. Host driver-595 baseline geomean `3.5108 ms` chậm hơn driver-580
`2.0355 ms` khoảng `72.48%`; optimized geomean `0.29745 ms` chậm hơn
`0.25752 ms` khoảng `15.51%`. Vì source revision không đổi, chênh headline
`7.904x → 11.803x` là cross-host ratio effect, không phải code improvement.

Artifacts curated: `results/timeline/`, promoted
`results/final/`, và archive cũ `results/archive/cross-host-driver580/`.

## 30. S8.17 — Repository structure cleanup

**Trạng thái:** Complete. Packaging and local smoke validation only; không có
algorithm change hoặc performance claim mới.

**Giả thuyết/phạm vi:** Tách active submission, candidate, runner, supplemental
docs, curated evidence và generated output sẽ làm import/reproduction rõ hơn mà
không đổi model behavior. Root giữ `main.py`, `v16_1_clean.py`, reference và
canonical docs; V19 chuyển vào `candidates/v19/`; runner chuyển vào `tools/` và
`tools/shape14/`; generated artifacts chuyển vào gitignored `runs/`.

**Validation:** `py_compile` PASS cho active source, toàn bộ tool, V19 package và
tests; module/direct-path list-shape/list-checkpoint smoke PASS; timeline
preflight V16.1 PASS strict weight equivalence. `python3 -m unittest discover
-s tests -v` PASS `5/5`, gồm `main` trỏ đúng standalone V16.1, repository-root
resolution, V19 imports, strict state-dict compatibility và causal/non-causal ×
mask/no-mask comparator. Official-shape #1 CPU one-trial diagnostic PASS
`0/1,048,576`; profile subprocess shape #2 CPU PASS. CPU timings chỉ kiểm tra
orchestration, không phải performance benchmark hợp lệ.

**Kết luận:** Active public interface và baseline không đổi. Module commands là
canonical; path execution vẫn được giữ tương thích bằng repository-root
bootstrap. Curated raw evidence chỉ đổi vị trí thư mục, nội dung machine-readable
không bị sửa.
