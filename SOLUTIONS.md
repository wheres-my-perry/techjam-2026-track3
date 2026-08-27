# Danh mục giải pháp và thử nghiệm

## 1. Cách dùng tài liệu này

Mỗi phương án cần ghi rõ giả thuyết, thay đổi, coverage correctness, command benchmark, môi trường và kết quả. Chỉ điền speedup sau khi có log đo thực tế.

Technical report tổng hợp cho các implementation hiện tại nằm tại [SOLUTION.md](SOLUTION.md). `SOLUTIONS.md` giữ vai trò experiment log; khi thuật toán hoặc candidate thay đổi, phải cập nhật cả hai file.

Public repository: [wheres-my-perry/techjam-2026-track3](https://github.com/wheres-my-perry/techjam-2026-track3).

Correctness gate mặc định: `relative error < 0.02 OR absolute error < 0.002` cho từng phần tử. Hai phép so sánh đều là strict `<`.

Trạng thái dùng trong tài liệu:

- **Idea**: chưa triển khai.
- **Implemented**: đã có code nhưng chưa đủ kết quả xác nhận.
- **Validated**: accuracy matrix đã pass trên môi trường ghi kèm.
- **Benchmarked**: đã có số đo hiệu năng tái lập được.
- **Rejected**: không đạt correctness, hiệu năng hoặc chi phí triển khai.

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

**Trạng thái:** GPU smoke benchmarked; full 14-shape matrix pending.

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
| FP32, non-causal, không padding | PASS GPU, exact | 1.035x ở default shape |
| FP32, causal, có padding | PASS local, exact | Chưa đo GPU |
| BF16, non-causal, có padding | PASS local, exact | Chưa đo GPU |
| FP16 | Chưa kiểm tra | Chưa đo GPU |

### Command mẫu

```bash
CUDA_VISIBLE_DEVICES=1 python3 v1_fuseQKV.py \
  --device cuda:0 \
  --dtype float32 \
  --batch-size 8 \
  --seq-len 128 \
  --d-model 512 \
  --heads 8 \
  --ffn-dim 2048 \
  --layers 6 \
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

GPU smoke benchmark sau khi import baseline chính thức:

```bash
cd /home/chim/techjam-2026-track3
source .venv/bin/activate
CUDA_VISIBLE_DEVICES=1 python v1_fuseQKV.py --device cuda:0 --dtype float32
```

Môi trường: GPU vật lý index `1`, NVIDIA GeForce RTX 5090; driver `595.58.03`; PyTorch `2.13.0+cu130`; FP32; TF32 bật; shape `(B=8, S=128, D=512, H=8, FFN=2048, L=6)`; non-causal; không padding; seed `1234`; warmup `20`; repeats `100`; rounds `3`.

| GPU | PyTorch/CUDA | Shape | Dtype/mask | Baseline median | Optimized median | Speedup | Accuracy |
|---|---|---|---|---:|---:|---:|---|
| RTX 5090, physical index 1 | 2.13.0+cu130 / 13.0 | 8×128×512, H=8, FFN=2048, L=6 | FP32, non-causal, no padding | 1.3569 ms | 1.3112 ms | 1.035x | PASS, max_abs=0, max_rel=0 |

Đây là smoke result cho default shape, chưa phải full-matrix result để nộp.

## 4. S2 — PyTorch SDPA

**Trạng thái:** Bản flattened đã PASS các nhánh GPU chính; full official-shape matrix pending.

**File:** `v2_SPDA.py`.

### Bản lịch sử dùng để đối chiếu

`v1_old.py` từng được phục dựng tạm thời để kiểm tra implementation đạt `1.635x` trên config mặc định cũ. File đã được xóa sau khi logic tốt hơn được merge vào `v2_SPDA.py`:

- Packed QKV được reshape/permute/unbind thành view, không gọi `.contiguous()` ba lần.
- Padding/causal mask được tạo một lần ngoài loop và dùng chung cho mọi layer.
- Non-causal dùng SDPA; causal giữ attention math reference.
- Training và dtype khác FP32 fallback toàn bộ về baseline.

GPU rebenchmark ngày 2026-08-27 dùng RTX 5090 vật lý index `1`, PyTorch `2.13.0+cu130`, FP32, TF32 bật, seed `1234`, warmup `20`, repeats `100`, rounds `3`.

| Candidate/config | Accuracy | Baseline median | Optimized median | Speedup |
|---|---|---:|---:|---:|
| `v1_old`, B8/S128/D512/H8/FFN2048/L6, non-causal | PASS, max_abs=0.000702024 | 1.3685 ms | 0.7676 ms | **1.783x** |
| `v2_SPDA` trước no-copy, cùng config | PASS, max_abs=0.000702024 | 1.3767 ms | 0.9951 ms | 1.384x |
| `v2_SPDA` no-copy, cùng config | PASS, max_abs=0.000702024 | 1.3747 ms | 0.8993 ms | 1.529x |
| `v2_SPDA` flattened, cùng config | PASS, max_abs=0.000662863 | 1.3758 ms | 0.7724 ms | **1.781x** |
| `v1_old`, official shape #1, causal | PASS exact | 1.0134 ms | 0.7982 ms | 1.270x |
| `v2_SPDA` trước no-copy, official shape #1, causal | PASS, max_abs=0.00105309 | 1.0120 ms | 0.7469 ms | 1.355x |
| `v2_SPDA` no-copy, official shape #1, causal | PASS, max_abs=0.00105309 | 1.0282 ms | 0.6874 ms | **1.496x** |
| `v2_SPDA` flattened, official shape #1, causal | PASS, max_abs=0.00105309 | 1.0558 ms | 0.5494 ms | **1.922x** |

Phép đo ablation cho thấy chênh lệch non-causal còn lại nằm chủ yếu ở Python/module dispatch mỗi layer: inline toàn bộ model đạt `1.786x`, còn chỉ inline block hoặc đổi cách reshape vẫn quanh `1.53x`. Vì vậy `v2_SPDA` dùng whole-model loop tương tự bản lịch sử, nhưng tiếp tục dùng SDPA cho cả causal và non-causal.

### Giả thuyết

SDPA giảm intermediate attention; fused QKV tiếp tục giảm ba GEMM projection và ba lần đọc cùng input xuống một GEMM.

### Phạm vi

- Gộp Q/K/V bằng một `F.linear` trên FP32 inference path.
- Reshape/permute/unbind packed QKV thành view để tránh ba lần `.contiguous()` mỗi layer.
- Inline attention trong loop model để bỏ `SDPASelfAttention.__call__` và helper dispatch mỗi layer.
- Refresh packed QKV sau `load_state_dict()`; training dùng projection gốc.
- Dùng `is_causal=True` khi không có padding mask.
- Kết hợp causal và valid-token mask bằng boolean mask khi có padding.
- Giữ output projection và zero-out padding query như baseline.
- FP16/BF16 tạm fallback về attention reference vì SDPA low-precision chưa đạt accuracy local.

### Local correctness

PyTorch `2.12.1`, CPU, strict `rtol=0.02` OR `atol=0.002`:

| Dtype | Causal | Padding | Kết quả |
|---|:---:|:---:|---|
| FP32 | off/on | off/on | PASS |
| BF16 | on | on | PASS, reference fallback |
| FP16 | on | on | PASS, reference fallback |

Default FP32 causal shape `(8, 128, 512, H=8, FFN=2048, L=6)` cũng PASS (`max_abs=2.15e-6`).

Số timing CPU smoke không dùng để đánh giá candidate.

### GPU benchmark sau khi flatten

```bash
cd /home/chim/techjam-2026-track3
source .venv/bin/activate
CUDA_VISIBLE_DEVICES=1 python v2_SPDA.py \
  --device cuda:0 --dtype float32 \
  --accuracy-trials 1 --warmup 20 --repeats 100 --benchmark-rounds 3
CUDA_VISIBLE_DEVICES=1 python v2_SPDA.py \
  --device cuda:0 --dtype float32 \
  --batch-size 64 --seq-len 128 --d-model 128 \
  --heads 4 --ffn-dim 128 --layers 4 --causal \
  --accuracy-trials 3 --warmup 20 --repeats 100 --benchmark-rounds 3
```

Môi trường: RTX 5090 vật lý index `1`; PyTorch `2.13.0+cu130`; TF32 bật; seed `1234`. Có một process khác giữ khoảng 3.9 GB VRAM nhưng báo 0% GPU utilization trước lượt đo; baseline và optimized vẫn chạy nối tiếp trong cùng process.

| Shape/dtype/mask | Accuracy | Baseline median | SDPA median | Speedup |
|---|---|---:|---:|---:|
| B8/S128/D512/H8/FFN2048/L6, FP32, non-causal | PASS, max_abs=0.000662863, failed=0/524288 | 1.3758 ms | 0.7724 ms | **1.781x** |
| B64/S128/D128/H4/FFN128/L4, FP32, causal | PASS, max_abs=0.00105309, failed=0/3145728 | 1.0558 ms | 0.5494 ms | **1.922x** |

So với bản no-copy còn gọi attention module riêng, flatten giảm optimized median `14.1%` ở non-causal (`0.8993` xuống `0.7724 ms`) và `20.1%` ở causal (`0.6874` xuống `0.5494 ms`). GPU validation bổ sung cũng PASS: causal + padding `max_abs=0.00105309`, non-causal + padding `max_abs=0.000714183`; BF16/FP16 fallback khớp exact. Các lượt padding/fallback dùng số repeats nhỏ chỉ để validate, không xem là benchmark chính thức.

## 5. S3 — Shape-aware scheduler

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

## 6. S4 — Low-precision optimized path

**Trạng thái:** Idea.

Mục tiêu là loại fallback toàn phần cho FP16/BF16 mà vẫn đạt tolerance. Candidate:

- Packed QKV với accumulation phù hợp.
- Chọn SDPA backend theo correctness đã đo.
- Giữ softmax hoặc một số reduction ở FP32.
- Dùng error-budget theo từng layer để tìm nguồn divergence.

Không promote phương án nếu chỉ pass shape nhỏ hoặc một seed.

## 7. S5 — FFN và LayerNorm fusion

**Trạng thái:** Idea.

Candidate:

- `torch.compile` để fuse elementwise/residual quanh LayerNorm và GELU.
- Triton kernel cho residual + LayerNorm.
- Fused Linear + activation khi backend hỗ trợ.
- Giảm intermediate allocation và memory round trip.

Cần profile trước; với model dimension lớn, GEMM có thể chiếm ưu thế và custom elementwise kernel không tạo speedup đáng kể.

## 8. S6 — Custom attention kernel

**Trạng thái:** Idea, chi phí cao.

Chỉ triển khai sau khi profiler chứng minh attention là bottleneck và SDPA không đủ tốt cho shape mục tiêu. Phải hỗ trợ causal/padding semantics và kiểm tra numerical stability, không chỉ kernel throughput riêng lẻ.

## 9. Ma trận benchmark đề xuất

Tối thiểu bao phủ:

- Batch: nhỏ và lớn.
- Sequence: ngắn, trung bình, dài.
- Hidden dimension/head count: tất cả tổ hợp ban tổ chức công bố.
- Dtype: FP32, FP16, BF16 nếu benchmark hỗ trợ.
- Causal: bật/tắt.
- Padding ratio: 0 và ít nhất một giá trị > 0.
- Nhiều seed và input scale.

Với mỗi case, lưu cả pass/fail, worst error, median, p90, throughput và speedup.
