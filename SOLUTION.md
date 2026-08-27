# Giải pháp tối ưu Transformer trên GPU

## 1. Tổng quan

Repository giải Track 3 của TikTok TechJam 2026 bằng PyTorch. Mục tiêu là giảm latency của một Transformer nhiều layer trên GPU trong khi giữ output tương đương reference theo luật của đề:

```text
relative_error < 0.02 OR absolute_error < 0.002
```

Hai phép so sánh đều dùng dấu `<` nghiêm ngặt và được kiểm tra trên từng phần tử output. Correctness là cổng bắt buộc: candidate chỉ được benchmark sau khi accuracy pass.

Hiện repository có hai implementation:

| Phiên bản | Tối ưu chính | Vai trò hiện tại |
|---|---|---|
| `v1_fuseQKV.py` | Gộp ba projection Q/K/V thành một `F.linear` | Bản tối ưu đơn giản, dùng để cô lập lợi ích của QKV fusion |
| `v2_SPDA.py` | Packed QKV không-copy + PyTorch SDPA + whole-model loop | Candidate FP32 nhanh nhất hiện tại |

`torch_transformer_benchmark.py` là baseline/reference và benchmark harness. Baseline không bị sửa để tạo speedup hoặc nới correctness.

## 2. Transformer reference

Input có shape:

$$
X \in \mathbb{R}^{B \times S \times D}
$$

Với mỗi Transformer block, reference chạy:

1. Pre-LayerNorm.
2. Ba projection riêng cho Query, Key và Value.
3. Scaled dot-product attention.
4. Output projection và residual add.
5. Pre-LayerNorm.
6. FFN `Linear → GELU → Linear` và residual add.
7. Zero-out padding query nếu có.

Self-attention được tính theo:

$$
Q=XW_Q,\quad K=XW_K,\quad V=XW_V
$$

$$
\operatorname{Attention}(Q,K,V)
=\operatorname{softmax}\left(\frac{QK^T}{\sqrt{d_h}}\right)V
$$

Reference dùng ba `nn.Linear` riêng, materialize attention scores, áp causal/padding mask, chạy softmax ở FP32 rồi nhân với V. Cách này rõ ràng và ổn định nhưng tạo nhiều kernel launch, đọc input lặp lại và materialize tensor attention trung gian.

## 3. Nguyên tắc triển khai chung

Các implementation tối ưu vẫn giữ:

- Giao diện `forward(x, valid_token_mask)`.
- Output shape `[batch_size, seq_len, d_model]`.
- Tên parameter và cấu trúc `state_dict` tương thích baseline.
- Causal semantics và key-padding semantics của reference.
- Output bằng 0 tại các padding query position.
- Cùng weights, input, seed, GPU, dtype, TF32, warmup và số vòng đo khi so sánh.

Packed weights được đăng ký là non-persistent buffers. Vì vậy chúng di chuyển cùng model khi gọi `.to(device, dtype)` nhưng không làm thay đổi public `state_dict`. Sau `load_state_dict()`, cache luôn được refresh từ ba projection gốc.

## 4. Solution 1 — Fused QKV projection

### 4.1 Vấn đề

Baseline chạy ba phép chiếu độc lập:

```text
Q = q_proj(X)
K = k_proj(X)
V = v_proj(X)
```

Ba phép chiếu có cùng input và cùng output dimension. Chúng tạo ba lần dispatch GEMM và đọc X ba lần.

### 4.2 Cách tối ưu

`v1_fuseQKV.py` ghép weights và biases theo output dimension:

$$
W_{QKV}=\operatorname{concat}(W_Q,W_K,W_V)
$$

$$
b_{QKV}=\operatorname{concat}(b_Q,b_K,b_V)
$$

Sau đó chỉ chạy:

```python
qkv = F.linear(x, packed_weight, packed_bias)
q, k, v = qkv.chunk(3, dim=-1)
```

Phần attention score, masking, FP32 softmax, output projection, residual và FFN được giữ nguyên. Nhờ vậy v1 cô lập đúng tác động của việc thay ba projection bằng một projection.

### 4.3 Lifecycle và training

- Packed weight/bias được tạo khi khởi tạo attention.
- Cache được refresh sau `load_state_dict()`.
- Khi `training=True`, implementation dùng lại ba projection gốc để tránh dùng cache stale sau optimizer step.
- Inference với weights cố định dùng packed projection.

### 4.4 Correctness và hiệu năng

Các smoke test local đã PASS exact cho FP32 non-causal, FP32 causal + padding và BF16 + padding.

GPU smoke benchmark:

| Cấu hình | Baseline | Fused QKV | Speedup | Accuracy |
|---|---:|---:|---:|---|
| B8/S128/D512/H8/FFN2048/L6, FP32, non-causal | 1.3569 ms | 1.3112 ms | 1.035x | PASS exact |

Kết quả cho thấy QKV fusion đơn lẻ có lợi nhưng chỉ giảm một phần nhỏ tổng latency. Attention core, tensor copies, Python/module dispatch, LayerNorm và FFN vẫn chưa được tối ưu.

## 5. Solution 2 — Packed QKV + SDPA + flattened model loop

`v2_SPDA.py` là candidate tốt nhất hiện tại. Tên file được giữ theo lịch sử repository; primitive được dùng là PyTorch Scaled Dot-Product Attention, viết tắt đúng là **SDPA**.

### 5.1 Packed QKV bằng view không-copy

V2 vẫn dùng một packed `F.linear`, nhưng không tách Q/K/V rồi gọi `.contiguous()` ba lần. Output packed được chuyển trực tiếp từ:

```text
[B, S, 3D]
    ↓ reshape
[B, S, 3, H, Dh]
    ↓ permute + unbind
Q, K, V: [B, H, S, Dh]
```

`reshape → permute → unbind` tạo các view phù hợp cho SDPA và loại ba explicit memory copies mỗi layer.

### 5.2 Dùng PyTorch SDPA

Thay vì tự materialize `QKᵀ`, softmax và phép nhân với V, v2 gọi:

```python
F.scaled_dot_product_attention(q, k, v, ...)
```

PyTorch có thể chọn backend SDPA phù hợp với GPU, dtype, shape và mask. Implementation vẫn truyền đúng scale `1 / sqrt(head_dim)`.

Các đường mask:

| Causal | Padding | Đường thực thi |
|:---:|:---:|---|
| Không | Không | SDPA không mask |
| Có | Không | `is_causal=True`, không tạo explicit causal mask |
| Không | Có | Boolean key mask `[B, 1, 1, S]` |
| Có | Có | Kết hợp key mask với lower-triangular mask `[S, S]` một lần ngoài layer loop |

Sau attention và sau mỗi block, invalid query positions được đưa về 0 giống baseline.

### 5.3 Flatten whole-model forward

Bản v2 ban đầu đặt SDPA trong một attention subclass riêng. Ablation trên GPU cho thấy phần chênh lệch còn lại chủ yếu đến từ Python/module dispatch mỗi layer:

| Biến thể ablation, non-causal default shape | Speedup |
|---|---:|
| Packed QKV no-copy nhưng giữ attention module riêng | khoảng 1.53x |
| Chỉ inline Transformer block | khoảng 1.53x |
| Đổi scale/reshape nhưng giữ attention module | khoảng 1.54x |
| Inline toàn bộ model loop | khoảng 1.79x |

Vì vậy v2 gọi packed projection, SDPA, output projection và FFN trực tiếp trong một whole-model loop. Thay đổi này không fuse GEMM bằng custom kernel; nó giảm Python dispatch và các khoảng trống giữa nhiều operator nhỏ.

### 5.4 FP32 optimized path và safe fallback

Đường SDPA tối ưu hiện chỉ dùng khi:

```text
model.eval() AND x.dtype == torch.float32
```

Training, FP16 và BF16 fallback toàn bộ về reference path. Đây là lựa chọn correctness-first: low-precision optimized path chưa được promote vì chưa có đủ accuracy evidence trên toàn bộ shape/seed. Fallback cho output exact so với baseline nhưng không tạo speedup đáng kể.

### 5.5 Cache packed weights

Mỗi attention layer vẫn giữ `q_proj`, `k_proj`, `v_proj` gốc nên weight copy từ baseline dùng strict mode. Hai buffers `_qkv_weight` và `_qkv_bias`:

- Không persistent trong `state_dict`.
- Được tạo bằng cách concatenate parameters gốc.
- Được refresh sau `load_state_dict()`.
- Chỉ được dùng trên inference path với weights cố định.

Nếu code bên ngoài sửa trực tiếp parameter trong eval mode mà không gọi `load_state_dict()`, cache cần được refresh thủ công. Benchmark hiện tại không có đường mutation này.

## 6. Correctness validation

Accuracy dùng luật elementwise:

```text
abs(user - reference) < 0.002
OR
abs(user - reference) < 0.02 * abs(reference)
```

Max relative error có thể rất lớn tại reference values gần 0 nhưng case vẫn đúng nếu absolute error nhỏ hơn `0.002`. Vì vậy số phần tử fail mới là điều kiện quyết định, không phải chỉ nhìn `max_rel`.

### 6.1 Local matrix

Môi trường local: PyTorch `2.12.1`, CPU.

| Dtype | Causal | Padding | Kết quả |
|---|:---:|:---:|---|
| FP32 | Tắt/bật | Không/có | PASS |
| BF16 | Bật | Có | PASS exact, reference fallback |
| FP16 | Bật | Có | PASS exact, reference fallback |

### 6.2 GPU matrix đã chạy

Môi trường: RTX 5090 vật lý index `1`, PyTorch `2.13.0+cu130`, CUDA `13.0`.

| Dtype/mask | Shape | Trials | Max abs | Failed elements | Kết quả |
|---|---|---:|---:|---:|---|
| FP32, non-causal, không padding | B8/S128/D512/H8/FFN2048/L6 | 3 | 0.000665478 | 0/1,572,864 | PASS |
| FP32, causal, không padding | B64/S128/D128/H4/FFN128/L4 | 3 | 0.00105309 | 0/3,145,728 | PASS |
| FP32, causal + padding | B64/S128/D128/H4/FFN128/L4 | 3 | 0.00105309 | 0/3,145,728 | PASS |
| FP32, non-causal + padding | B8/S128/D512/H8/FFN2048/L6 | 3 | 0.000714183 | 0/1,572,864 | PASS |
| FP16, causal + padding | B8/S128/D128/H4/FFN128/L4 | 2 | 0 | 0/262,144 | PASS, fallback |
| BF16, causal + padding | B8/S128/D128/H4/FFN128/L4 | 2 | 0 | 0/262,144 | PASS, fallback |

Đây chưa phải full matrix của 14 shape chính thức. Các extreme cases như batch `10000` hoặc sequence length `100000` chưa được xác nhận.

## 7. GPU benchmark

### 7.1 Môi trường

| Thành phần | Giá trị |
|---|---|
| OS | Debian GNU/Linux, kernel `6.12.74+deb13+1-amd64` |
| GPU | NVIDIA GeForce RTX 5090, physical index `1` |
| PCI bus | `0000:41:00.0` |
| Driver | `595.58.03` |
| Python environment | `/home/chim/techjam-2026-track3/.venv` |
| PyTorch | `2.13.0+cu130` |
| CUDA | `13.0` |
| Dtype | FP32 |
| TF32 | Bật cho baseline và optimized |
| Seed | `1234` |
| Warmup/repeats/rounds | `20 / 100 / 3` |

GPU được cô lập bằng `CUDA_VISIBLE_DEVICES=1`; do đó PyTorch nhìn GPU vật lý số 1 dưới tên logic `cuda:0`.

Trước lượt đo có một process khác giữ khoảng 3.9 GB VRAM nhưng báo 0% utilization. Baseline và optimized vẫn được đo nối tiếp trong cùng process với thứ tự đảo theo round. Kết quả hiện tại đủ để so sánh candidate, nhưng nên chạy lại trên GPU hoàn toàn idle trước submission cuối.

### 7.2 Kết quả v2

| Config | Baseline median / p90 | V2 median / p90 | V2 throughput | Speedup |
|---|---:|---:|---:|---:|
| B8/S128/D512/H8/FFN2048/L6, non-causal | 1.3758 / 1.4028 ms | 0.7724 / 0.7949 ms | 1,325,820 token/s | **1.781x** |
| B64/S128/D128/H4/FFN128/L4, causal | 1.0558 / 1.1387 ms | 0.5494 / 0.5971 ms | 14,911,463 token/s | **1.922x** |

So với bản v2 no-copy nhưng chưa flatten:

- Non-causal giảm từ `0.8993` xuống `0.7724 ms`, tương đương giảm latency `14.1%`.
- Causal giảm từ `0.6874` xuống `0.5494 ms`, tương đương giảm latency `20.1%`.

## 8. Cách tái lập

### 8.1 Chuẩn bị môi trường

Trên máy GPU hiện tại:

```bash
cd /home/chim/techjam-2026-track3
source .venv/bin/activate
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Với môi trường mới, tạo venv riêng và cài PyTorch build tương thích CUDA/GPU của máy. Không dùng Python hệ thống nếu environment đó chưa có PyTorch.

### 8.2 Kiểm tra syntax

```bash
python -m py_compile torch_transformer_benchmark.py v1_fuseQKV.py v2_SPDA.py
```

### 8.3 Benchmark v1

```bash
CUDA_VISIBLE_DEVICES=1 python v1_fuseQKV.py \
  --device cuda:0 --dtype float32
```

### 8.4 Benchmark v2 non-causal

```bash
CUDA_VISIBLE_DEVICES=1 python v2_SPDA.py \
  --device cuda:0 --dtype float32 \
  --accuracy-trials 1 \
  --warmup 20 --repeats 100 --benchmark-rounds 3
```

### 8.5 Benchmark v2 trên official shape #1 causal

```bash
CUDA_VISIBLE_DEVICES=1 python v2_SPDA.py \
  --device cuda:0 --dtype float32 \
  --batch-size 64 --seq-len 128 --d-model 128 \
  --heads 4 --ffn-dim 128 --layers 4 --causal \
  --accuracy-trials 3 \
  --warmup 20 --repeats 100 --benchmark-rounds 3
```

Thêm `--padding-ratio 0.25` để kiểm tra nhánh padding. Đổi `--dtype` thành `float16` hoặc `bfloat16` để xác nhận reference fallback.

## 9. Vì sao v2 nhanh hơn v1

| Chi phí | Baseline | V1 | V2 |
|---|:---:|:---:|:---:|
| Ba Q/K/V GEMM riêng | Có | Không | Không |
| Ba explicit contiguous head copies | Có | Có | Không |
| Materialize attention scores bằng code reference | Có | Có | Không, dùng SDPA |
| Tạo causal mask trong từng layer | Có | Có | Không |
| Attention module dispatch mỗi layer | Có | Có | Không trên optimized path |
| FP16/BF16 optimized path | Không | Packed QKV | Chưa, fallback reference |

V1 chỉ tối ưu một phần nhỏ của block nên đạt `1.035x`. V2 cộng dồn QKV fusion, layout view, SDPA, mask reuse và giảm dispatch nên đạt `1.781x–1.922x` trên hai config đã đo.

## 10. Công cụ và quy trình phát triển

- PyTorch được dùng cho reference, optimized implementation, correctness và CUDA Event timing.
- OpenAI Codex được dùng để đối chiếu đề với benchmark, phân tích code, đề xuất và triển khai candidate, chạy test local/GPU và thực hiện ablation giữa các biến thể.
- Input benchmark là tensor sinh ngẫu nhiên với seed cố định; project không dùng dataset bên ngoài.
- Mỗi thay đổi được kiểm tra syntax và accuracy trước khi benchmark hiệu năng.
- Baseline và optimized luôn nhận cùng weights và cùng input trong một process.

## 11. Giới hạn hiện tại

1. Chưa chạy đủ 14 official shapes, nhiều seed và nhiều input scale.
2. V2 chỉ tối ưu FP32 inference; FP16/BF16 và training dùng reference fallback.
3. Chưa có matrix runner hoặc JSON/CSV artifact để chạy và lưu toàn bộ kết quả tự động.
4. Chưa profile bằng PyTorch Profiler hoặc Nsight; kết luận về dispatch dựa trên ablation timing.
5. Chưa có shape-aware scheduler; một implementation duy nhất khó tối ưu cho cả batch/sequence rất nhỏ và rất lớn.
6. Chưa triển khai custom Triton/CUDA kernel, FFN fusion hoặc residual + LayerNorm fusion.
7. Packed cache cần refresh nếu weights bị sửa trực tiếp trong eval mode.
8. CPU model và disk metadata chưa được lưu; cần bổ sung trước technical report cuối.

## 12. Hướng phát triển tiếp theo

Ưu tiên theo thứ tự:

1. Chạy full correctness matrix trên 14 official shapes.
2. Thêm benchmark matrix runner và xuất JSON/CSV cùng metadata môi trường/git revision.
3. Profile QKV, SDPA, output projection, LayerNorm và FFN theo từng nhóm shape.
4. Xây FP16/BF16 optimized path có kiểm soát error budget.
5. Thử `torch.compile`, residual + LayerNorm fusion và FFN/GELU fusion.
6. Chỉ triển khai Triton/custom CUDA khi profiler chứng minh PyTorch SDPA là bottleneck.
7. Tạo scheduler theo `(GPU, dtype, B, S, D, H, causal, padding)` và luôn có safe fallback.

## 13. Kết luận

Packed QKV đơn lẻ là optimization an toàn nhưng lợi ích hạn chế. Candidate v2 giải quyết thêm hai nguồn chi phí lớn: attention intermediates và per-layer dispatch. Trên RTX 5090 được chỉ định, v2 đạt **1.781x** ở config non-causal mặc định và **1.922x** ở official shape #1 causal, đồng thời pass strict correctness trên các nhánh FP32 causal/non-causal và padding đã kiểm tra.

V2 là implementation nên tiếp tục phát triển. V1 vẫn có giá trị như một ablation dễ đọc và fallback thử nghiệm, nhưng chưa phải candidate cuối để nộp.
