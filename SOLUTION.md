# Giải pháp tối ưu Transformer trên GPU

## 1. Tổng quan

Repository giải Track 3 của TikTok TechJam 2026 bằng PyTorch. Mục tiêu là giảm latency của một Transformer nhiều layer trên GPU trong khi giữ output tương đương reference theo luật của đề:

```text
relative_error < 0.02 OR absolute_error < 0.002
```

Hai phép so sánh đều dùng dấu `<` nghiêm ngặt và được kiểm tra trên từng phần tử output. Correctness là cổng bắt buộc: candidate chỉ được benchmark sau khi accuracy pass.

Hiện repository có chuỗi implementation phiên bản hóa cùng execution configuration compiled:

| Phiên bản | Tối ưu chính | Vai trò hiện tại |
|---|---|---|
| `main.py` | Benchmark wrapper import standalone V16.1 clean | Final benchmark entrypoint; #1–#14 strict PASS |
| `v16_1_clean.py` | Toàn bộ V16.1 model/config/kernel/cache/executor trong một file | Active standalone artifact; driver-595 #1–#13 geomean 11.803x |
| `v1_fuseQKV.py` | Gộp ba projection Q/K/V thành một `F.linear` | Bản tối ưu đơn giản, dùng để cô lập lợi ích của QKV fusion |
| `v2_SPDA.py` | V1 + PyTorch SDPA | Ablation đo riêng tác động của SDPA |
| `v3_SDPA_NoCopy.py` | Packed QKV không-copy + SDPA + whole-model loop | Ablation trước v3.1 |
| `v3_1_CausalMask.py` | V3 + causal flag trực tiếp + một padding zero mỗi block | Historical FP32 ablation |
| V3.1 + `torch.compile` | Whole-model Inductor compile; `reduce-overhead` ablation | Historical compile ablation trên official shape #1 |
| `v4_FP16.py` | FP16 internal GEMM+SDPA, FP32 norm/residual/output | Shape #1 PASS |
| `v4_1_FP16_GELU.py` | V4 FP16 + GELU trực tiếp FP16 để bỏ conversion kernels | Eager ablation PASS; compiled trùng hiệu năng V4 |
| `v4_2_SDPA_Dispatch.py` | V4.1 + static per-shape cuDNN/automatic SDPA dispatch | PASS #1–#13; geomean 7.58x, shape #14 pending |
| `v4_3_Flash.py` | Causal right-padding mask elision + Flash-first với cuDNN/Efficient/Math fallback | PASS #1–#13; max-autotune geomean 9.53x, shape #14 pending |
| `v4_3_flash_clean.py` | Standalone V4.3 config/model, không benchmark dependency | Strict state dict và local graph-equivalence smoke PASS |
| `v5_1_FP16Accum.py` | V4.3 + full FP16 accumulation cho eligible CUDA GEMM | Negative ablation: accuracy fail, không promote |
| `v6_ApproxGELU.py` | V4.3 + tanh-approximated FP16 GELU | Accuracy PASS #1–#13; paired gain nằm trong noise, không promote |
| `v7_ResidualLayerNorm.py` | V4.3 + pipelined residual/LayerNorm boundary | Correctness gate PASS; compiled graph trùng V4.3, không promote |
| `v8_FusedFFNGELU.py` | V4.3 + Triton FFN-in GEMM/bias/exact-GELU có shape dispatch | PASS #1–#13; #6 giảm `4.76%` latency so với V4.3, shape khác fallback |
| `v8_1_FusedFFNGELUAll.py` | V8 force custom FFN/GELU trên mọi shape | PASS #1–#13; geomean giảm khoảng 1%, nhưng có per-shape regression/noise nên không promote |
| `v9_PersistentMLP.py` | Fully fused persistent FFN-in/exact-GELU/FFN-out | Isolated win 1.18–1.59x và PASS #1–#13; whole-model không thắng ổn định, không promote |
| `v11_FP32PreGELU.py` | V8.1 + exact GELU trực tiếp từ FP32 FFN-in accumulator | Arithmetic path/rollback; GPU #1–#13 PASS |
| `v12_FP32FFNOut.py` | V11 + FFN-out GEMM store trực tiếp FP32 | Experimental ablation; local gate PASS, GPU accuracy/performance pending |
| `v12_1_FP32OutProj.py` | V11 + attention out-projection store trực tiếp FP32 | Experimental ablation; local gate PASS, GPU pending |
| `v12_2_FP32ResidualOutputs.py` | V11 + cả hai residual-branch projection output FP32 | Experimental ablation; local gate PASS, GPU pending |
| `v13_INT8FFNProbe.py` | V11 + accuracy-only symmetric INT8 FFN-in simulation | Negative ablation: official #2 FAIL cả W8/A8/W8A8; không viết kernel/benchmark |
| `v14_BatchChunked.py` | V11 + exact batch-independent chunking cho shape #14 | Strict #14 PASS `0/3.2768B`; optimized-only median `6683.9873 ms`, paired speedup N/A |
| `v14_1_BatchChunked.py` | V11 + large-sequence cutoff dispatcher | Parent/rollback; `S < 8192` dùng V11, FP32 eval `S >= 8192` batch-chunk |
| `v15_DirectQKVLayout.py` | V14.1 + exact-#13 Triton QKV ghi thẳng layout cho Flash | QKV parent/rollback; paired #13 win hai orders |
| `v15_1_DirectQKVAll.py` | V15 force direct-layout QKV cho causal `S<8192` | Cross-shape ablation; #1–#12 PASS, chỉ #6 thắng ổn định, không promote force-all |
| `v16_CompiledBatchExecutor.py` | V15 + compiled/reused B=1 executor trong eager loop #14 | Previous main/direct-QKV rollback; full #14 strict PASS và latency giảm `3.11–3.61%` |
| `v16_1_NoDirectQKV13.py` | Historical composed V16.1, nay trong archive | Superseded về packaging bởi `v16_1_clean.py` |
| `v17_CompiledBatch2.py` | V16 + compiled executor B=2 cho #14 | Experimental; full strict PASS, gain chỉ `0.30–0.59%`, không promote |
| `v17_sage.py` | V16.1 + corrected SageAttention cross-shape | Negative ablation; full #1–#13 fail strict ở #6/#9 |
| `v18_sage.py` | V16.1 + direct automatic SageAttention SM120 | Performance-only diagnostic; không correction, không đổi main |
| `v19_CUDAFP16Checkpoint.py` | V16.1 + CUDA WMMA FP16 accumulate, checkpoint partial sum sang FP32 theo K | GPU PASS correctness nhưng regress; không promote |
| `v19_1_0_ParallelBatchV161.py` | V16.1 + multi-stream parallel partitions cho large batch | P4 measured winner #14; full strict PASS, chưa đổi main |
| `v19_1_1_ParallelBatchV19.py` | V19 arithmetic + cùng multi-stream batch scheduler | K64/P2 PASS nhưng chậm hơn V19.1.0 P4 |
| `shape14_accuracy.py` | Query-blocked reference + streaming strict comparator | Accuracy-only harness cho #14; không dùng để claim baseline latency |
| `shape14_optimized_benchmark.py` | CUDA Event optimized-only diagnostic | Tái lập latency #14 khi original baseline cần score ~18.6 TiB |
| `v4_1_clean.py` | Standalone V4.1 config/model, không benchmark dependency | Import/presentation artifact; graph-equivalent local validation PASS |

`torch_transformer_benchmark.py` là baseline/reference và benchmark harness. Baseline không bị sửa để tạo speedup hoặc nới correctness.

Final active-main result trên commit `4f77a04`: official #1–#13 strict PASS với
predeclared driver-595 start-control geomean `11.803x`; full #14 strict PASS
`0/3,276,800,000`, native B32 PASS và optimized-only median `6987.4644 ms`.
Environment và raw evidence nằm trong `results/final/`; driver-580 evidence
`7.904x` được giữ làm cross-host archive, không phải code baseline cho ratio mới.

Từ D-042, toàn bộ implementation lịch sử nằm trong `archive/versions/`. Root
giữ final `v16_1_clean.py` cùng experimental V19 mới; V16.1 không import
benchmark harness hoặc version cũ, còn V19 kế thừa nó có chủ đích cho ablation.
Việc archive không thay đổi các kết quả lịch sử được trình bày bên dưới.

Performance benchmark chỉ được xem là chính thức khi chạy đúng một trong 14 test shapes ở Appendix của đề. Shape khác vẫn có thể dùng cho correctness hoặc ablation, nhưng phải ghi rõ là **non-official diagnostic**.

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

GPU smoke benchmark lịch sử trên default shape (**non-official diagnostic**):

| Cấu hình | Baseline | Fused QKV | Speedup | Accuracy |
|---|---:|---:|---:|---|
| B8/S128/D512/H8/FFN2048/L6, FP32, non-causal, diagnostic | 1.3569 ms | 1.3112 ms | 1.035x | PASS exact |

Kết quả diagnostic cho thấy QKV fusion đơn lẻ có lợi nhưng chỉ giảm một phần nhỏ tổng latency. Attention core, tensor copies, Python/module dispatch, LayerNorm và FFN vẫn chưa được tối ưu. V1 chưa có official-shape performance result.

## 5. Solution 2 — V1 + PyTorch SDPA

`v2_SPDA.py` lấy v1 làm nền và chỉ thay explicit attention math bằng PyTorch Scaled Dot-Product Attention, viết tắt đúng là **SDPA**. Tên file `SPDA` được giữ để tương thích với lịch sử repository.

### 5.1 Phạm vi thay đổi

V2 giữ nguyên:

- Packed QKV `F.linear` và cache lifecycle của v1.
- `chunk(3)`, ba lần `_split_heads(...).contiguous()` cho Q/K/V.
- Attention module dispatch tại từng layer.
- Output projection, residual, FFN và zero-out padding query.

Thay đổi duy nhất trên FP32 path là gọi:

```python
F.scaled_dot_product_attention(q, k, v, ...)
```

thay cho explicit `QKᵀ → mask → FP32 softmax → probs @ V`.

### 5.2 Mask và fallback

- Non-causal không padding: SDPA không mask.
- Causal không padding: `is_causal=True`.
- Có padding: boolean key mask được tạo trong attention module; causal mask được kết hợp khi cần.
- FP16/BF16 giữ attention math của v1 để bảo toàn correctness.
- Training dùng ba projection gốc nên packed cache không bị dùng stale.

### 5.3 Kết quả

| Loại | Config | Baseline | V2 | Speedup | Accuracy |
|---|---|---:|---:|---:|---|
| Official shape #1 | B64/S128/D128/H4/FFN128/L4, FP32, causal | 1.0120 ms | 0.7469 ms | **1.355x** | PASS, max_abs=0.00105309 |
| Non-official diagnostic | B8/S128/D512/H8/FFN2048/L6, FP32, non-causal | 1.3767 ms | 0.9951 ms | 1.384x | PASS |

V2 có chủ đích chưa thêm no-copy views, mask reuse hoặc flattened loop. Nhờ vậy chênh lệch v1 → v2 đại diện tương đối rõ cho lợi ích của SDPA.

## 6. Solution 3 — Packed QKV + SDPA + flattened model loop

`v3_SDPA_NoCopy.py` là ablation trước V3.1. Nó giữ SDPA của v2 và tối ưu thêm data movement cùng Python/module dispatch.

### 6.1 Packed QKV bằng view không-copy

V3 vẫn dùng một packed `F.linear`, nhưng không tách Q/K/V rồi gọi `.contiguous()` ba lần. Output packed được chuyển trực tiếp từ:

```text
[B, S, 3D]
    ↓ reshape
[B, S, 3, H, Dh]
    ↓ permute + unbind
Q, K, V: [B, H, S, Dh]
```

`reshape → permute → unbind` tạo các view phù hợp cho SDPA và loại ba explicit memory copies mỗi layer.

### 6.2 Dùng PyTorch SDPA

Thay vì tự materialize `QKᵀ`, softmax và phép nhân với V, v3 gọi:

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

### 6.3 Flatten whole-model forward

V2 đặt SDPA trong một attention subclass riêng. Ablation trên GPU cho thấy phần chênh lệch còn lại chủ yếu đến từ Python/module dispatch mỗi layer:

| Biến thể ablation, non-causal default shape | Speedup |
|---|---:|
| Packed QKV no-copy nhưng giữ attention module riêng | khoảng 1.53x |
| Chỉ inline Transformer block | khoảng 1.53x |
| Đổi scale/reshape nhưng giữ attention module | khoảng 1.54x |
| Inline toàn bộ model loop | khoảng 1.79x |

Vì vậy v3 gọi packed projection, SDPA, output projection và FFN trực tiếp trong một whole-model loop. Thay đổi này không fuse GEMM bằng custom kernel; nó giảm Python dispatch và các khoảng trống giữa nhiều operator nhỏ.

### 6.4 FP32 optimized path và safe fallback

Đường SDPA tối ưu hiện chỉ dùng khi:

```text
model.eval() AND x.dtype == torch.float32
```

Training, FP16 và BF16 fallback toàn bộ về reference path. Đây là lựa chọn correctness-first: low-precision optimized path chưa được promote vì chưa có đủ accuracy evidence trên toàn bộ shape/seed. Fallback cho output exact so với baseline nhưng không tạo speedup đáng kể.

### 6.5 Cache packed weights

Mỗi attention layer vẫn giữ `q_proj`, `k_proj`, `v_proj` gốc nên weight copy từ baseline dùng strict mode. Hai buffers `_qkv_weight` và `_qkv_bias`:

- Không persistent trong `state_dict`.
- Được tạo bằng cách concatenate parameters gốc.
- Được refresh sau `load_state_dict()`.
- Chỉ được dùng trên inference path với weights cố định.

Nếu code bên ngoài sửa trực tiếp parameter trong eval mode mà không gọi `load_state_dict()`, cache cần được refresh thủ công. Benchmark hiện tại không có đường mutation này.

### 6.6 V3.1 — Causal flag và single padding zero

`v3_1_CausalMask.py` giữ nguyên packed QKV, no-copy views, SDPA và whole-model loop của v3. Hai thay đổi duy nhất là:

1. Truyền key-padding mask `[B,1,1,S]` trực tiếp vào SDPA cùng `is_causal=config.causal`, không tạo lower-triangular buffer hoặc tensor kết hợp `[B,1,S,S]`.
2. Bỏ zero-out attention output; invalid query chỉ được zero cuối block và sau final LayerNorm. FFN xử lý độc lập theo token nên giá trị invalid trung gian không ảnh hưởng token hợp lệ trước điểm zero này.

Training và dtype khác FP32 vẫn fallback toàn bộ về reference. V3 được giữ nguyên để làm ablation.

### 6.7 Whole-model compilation

V3.1 có thể được bọc sau weight copy, device transfer và `eval()` bằng:

```python
optimized = torch.compile(optimized, mode="reduce-overhead")
```

`torch.compile` dùng TorchInductor làm backend mặc định. Inductor tối ưu whole-model graph và có thể sinh fused Triton kernels; cấu hình `reduce-overhead` yêu cầu đường giảm dispatch/kernel-launch overhead bằng CUDA Graph khi graph đủ điều kiện. Compilation và CUDA Graph warmup xảy ra trước CUDA Event benchmark, nên con số dưới đây là steady-state inference latency chứ không phải cold-start latency.

Trên official shape #1, cùng RTX 5090/PyTorch `2.13.0+cu130`, accuracy trials `3`, warmup `50`, repeats `200`, rounds `5`, v3.1 compiled PASS strict correctness và giảm median từ `0.5118` xuống `0.3169 ms`, tương đương `38.1%`; overall speedup so với eager baseline tăng từ `2.112x` lên `3.404x`. Baseline của hai process chỉ lệch `0.19%`.

Không quy gain này trực tiếp cho LayerNorm. Eager ATen stage profile và compiled graph là hai attribution schema khác nhau: một fused Triton kernel có thể bao phủ nhiều ATen operation, còn CUDA Graph có thể giảm launch gaps mà không đổi từng operator. Vì vậy report tách ablation `default` versus `reduce-overhead` và dùng raw GPU/runtime trace để chứng minh kernel count, compiled regions, Triton/CUDA Graph launches và device time.

### 6.8 V4 mixed-precision internal compute

V4 thử giảm precision có chọn lọc thay vì đổi toàn bộ model dtype. Public weights,
input, LayerNorm, residual accumulation, GELU và output giữ FP32. Packed QKV,
SDPA, attention output projection và hai FFN GEMM dùng cache FP16
(`v4_FP16.py`) hoặc BF16 (`v4_BF16.py`), sau đó cast về FP32 trước residual add.

Cache low precision không persistent nên state dict vẫn strict-compatible với
baseline. Cache được refresh sau `load_state_dict()` và sau module device/dtype
transform; training hoặc input không phải FP32 dùng full reference fallback.
Hai file chỉ là experimental candidates cho đến khi pass strict accuracy trên
GPU và đủ official shapes. Chưa có latency/speedup nào được ghi nhận cho V4.

Local CPU diagnostic trên official shape #2 (ba trial) cho thấy V4 FP16 PASS với
`max_abs=0.00115311`, `failed=0/49,152`; V4 BF16 FAIL với
`max_abs=0.00808978`, `failed=1,569/49,152`, nên benchmark của BF16 được accuracy
gate bỏ qua. FP16 cũng PASS causal/non-causal × padding/no-padding trên ba seed và
PASS local `torch.compile(mode="default")` smoke.

Trên GPU official shape #1, V4 FP16 eager PASS 5 trial và đạt `0.5836 ms`
(`1.764x`) trong paired profiler run; V4 BF16 FAIL accuracy nên không có
performance result hợp lệ. V4 FP16 giảm GEMM và SDPA device time nhưng tăng
`aten::copy_` từ 5 calls ở V3.1 lên 29 calls, khiến eager path vẫn chậm hơn V3.1.

### 6.9 V4.1 — GELU trực tiếp FP16

`v4_1_FP16_GELU.py` giữ nguyên FP32 LayerNorm, residual và output của V4 nhưng
không cast FFN hidden lên FP32 trước GELU rồi xuống FP16 trước FFN-out. Trên model
bốn layer, ablation bỏ tám `aten::copy_`/kernel launches. Official shape #1 vẫn
PASS 5 trial với cùng summary như V4 (`max_abs=0.00159800`, `failed=0/5,242,880`).

Eager profiler xác nhận copy calls `29 → 21`, layout-copy self time `0.0695 →
0.0499 ms`, GPU kernels `100 → 92`; median giảm `0.5836 → 0.5492 ms` (`5.9%`).
Candidate vẫn chậm hơn V3.1 eager `0.5136 ms`, nên cast quanh QKV, attention
output và FFN boundaries vẫn là chi phí đáng kể.

Khi dùng `torch.compile(mode="reduce-overhead")`, V4 và V4.1 đều đạt `0.1858 ms`
và có cùng 34 GPU kernels/14 Triton device events. Inductor đã fuse conversion
quanh GELU, nên manual GELU-FP16 không tạo thêm compiled speedup. Điểm chính của
thí nghiệm là mixed precision + whole-model compile: `0.1858 ms` nhanh hơn V3.1
compiled `0.3168 ms` khoảng `41.4%` trong cùng run.

V4.1 eager còn PASS selected official shapes #2/#7/#8/#12/#13; shape #13 đạt
`2.7813 ms`, `15.013x`. Full 14-shape matrix và nhiều seed/input scale vẫn là
điều kiện trước promotion; shape #7 hiện có max absolute error `0.00188218`, khá
gần ngưỡng absolute dù toàn bộ phần tử vẫn pass luật OR.

`v4_1_clean.py` là bản standalone của cùng thuật toán: chỉ export
`TransformerConfig` và `UserOptimizedTransformer`, không import harness hoặc có
CLI. Parameter/state-dict names, FP16 cache refresh sau `load_state_dict()`/`to`,
optimized forward và fallback đều được giữ. Local equivalence test cho eager,
training và BF16 fallback khớp exact V4.1; compiled smoke sai khác tối đa
`4.76837e-07`, vẫn pass strict competition comparator. GPU RTX 5090 smoke cũng
khớp eager exact và compiled PASS với `max_abs=2.38419e-07`. Những smoke tests
này xác nhận packaging equivalence, không tạo performance result mới.

### 6.10 V4.2 — SDPA backend dispatch theo shape

Compiled trace của V4.1 dùng memory-efficient `fmha_cutlassF_f16`. V4.2 cô lập
automatic, Flash, Efficient và cuDNN SDPA trên cùng FP16 graph. Flash không hỗ
trợ non-null attention mask; forced Efficient khớp automatic; cuDNN nhanh hơn ở
một số shape nhưng chậm hơn hoặc không hỗ trợ shape khác. Vì vậy
`v4_2_SDPA_Dispatch.py` ép cuDNN chỉ cho official
#1/#2/#3/#4/#7/#9/#13; mọi config khác dùng automatic fallback.

Dispatch key được tính từ `(B,S,D,H,L,FFN,causal)` khi model khởi tạo. Đây là
Python constant trước `torch.compile`, nên compiled hot path không thêm tensor
condition, host sync hoặc kernel launch. State dict, FP16 caches, LayerNorm,
residual, GELU và masking semantics giữ nguyên V4.1.

Trên shape #13, cuDNN giảm attention kernel `1.2878 → 0.9590 ms` và candidate
median `2.3126 → 1.6841 ms` (`27.2%`). Shape #1 giảm `0.1858 → 0.1716 ms`
(`7.7%`). Matrix dispatcher #1–#13 PASS toàn bộ với geomean speedup `7.58x`, so
với khoảng `7.09x` của V4.1. Shape #14 chưa được tính và vẫn pending.

### 6.11 V4.3 — Causal key-mask elision và Flash-first fallback

Harness tạo `valid_token_mask` toàn `True` khi không padding, nên V4.2 vẫn truyền
non-null `attn_mask` và PyTorch Flash không chạy. V4.3 dùng invariant mạnh hơn:
với causal self-attention và right padding, valid tokens là một prefix; một valid
query chỉ nhìn các key ở vị trí trước hoặc bằng nó, do đó không thể nhìn padded
keys trong suffix. Invalid query outputs vẫn được zero cuối block. Key-padding
attention mask vì thế dư trên nhánh này.

`v4_3_Flash.py` không còn hard-code official shapes. Mọi causal optimized path
truyền `attn_mask=None` và khai báo backend priority Flash → cuDNN → Efficient →
Math bằng `sdpa_kernel(..., set_priority=True)`. PyTorch bỏ qua backend không hỗ
trợ device/shape. Non-causal giữ key mask và automatic dispatch; training hoặc
public input khác FP32 về full-reference fallback. Hot path không có
`.all().item()`, tensor condition hoặc host synchronization. Fast path yêu cầu
mask dạng prefix-true/suffix-false; arbitrary sparse mask không được phép suy ra
bằng causal invariant này.

Profiler xác nhận `pytorch_flash::flash_fwd_kernel`. Trên #13, attention kernel
giảm `0.9607 → 0.4232 ms` và end-to-end giảm `1.6843 → 1.1401 ms` so với V4.2;
#1 giảm `0.1735 → 0.1489 ms`. Direct matrix #1–#13 PASS strict accuracy và đạt
geomean `8.52x`. Static V4.3 dispatcher đạt `8.48x`; Flash-first chịu regression
#6 khoảng `2.5%` nhưng đơn giản hơn và không giảm aggregate score.

`v4_3_flash_clean.py` là bản standalone của cùng thuật toán: file chỉ export
`TransformerConfig` và `UserOptimizedTransformer`, không import harness hoặc có
CLI. Parameter names/state dict, FP16 cache refresh sau `load_state_dict()`/`to`,
Flash-first causal path, non-causal masked path và full-reference fallback đều
được giữ. Local equivalence smoke xác nhận strict state dict, cache dtype, eager
causal/non-causal × padding/no-padding, training và BF16 fallback đều khớp
`v4_3_Flash.py` exact. Strict local accuracy còn PASS official shape #2 qua ba
trial (`max_abs=0.00105053`) cùng causal/non-causal right-padding diagnostics;
compiled smoke PASS (`max_abs=0.000219792` so với eager). Đây là packaging
equivalence/correctness smoke, không tạo GPU benchmark mới.

### 6.12 V5 — FP8 và MXFP8 negative ablations

Phần này là hồ sơ của thử nghiệm đã loại; toàn bộ source V5 đã được xóa khỏi
working tree và không còn là implementation có thể chọn trong runner.

Target RTX 5090/PyTorch `2.13.0+cu130` có E4M3 scaled GEMM nhưng FP8 là shell
dtype, không thể truyền trực tiếp vào SDPA; attention vì thế vẫn chạy FP16.
`v5_FP8.py` cache weight E4M3 với per-tensor scale và quantize activation động
cho packed QKV, attention-output và hai FFN GEMM. Official shape #2 FAIL 3/3
trial (`max_abs=0.110912`, `25,429/49,152` phần tử fail), nên performance bị
accuracy gate bỏ qua.

`v5_1_MXFP8.py` dùng native Blackwell MXFP8 chính xác hơn về dynamic range: mỗi
block 32 giá trị E4M3 có E8M0 power-of-two scale; scale được pad và swizzle sang
layout 32×4×4 trước public `F.scaled_mm`. All-ones GEMM sanity trả đúng `128`,
nhưng full model vẫn FAIL (`max_abs=0.144848`, `25,800/49,152` fail). Các file
`v5_2_MXFP8_*.py` giữ chỉ một scope FP8 qua bốn layer; QKV-only là nhẹ nhất vẫn
FAIL `7,532/49,152` phần tử qua ba trial. Attention-output-only, FFN-input-only
và FFN-output-only cũng fail ngay trial đầu.

Row-wise FP8 không được dùng vì target runtime trả `3.5` thay vì `128` trên
all-ones sanity GEMM. Vì không V5 variant nào qua correctness, report không ghi
latency/speedup cho chúng. Kết quả này tách rõ “hardware có FP8” khỏi “model đạt
error budget”; hướng FP8 chỉ nên mở lại với calibration/QAT hoặc recipe khác đã
pass strict validation.

### 6.13 V5.1 — Full FP16 accumulation negative ablation

`v5_1_FP16Accum.py` giữ nguyên V4.3 nhưng bật process-global
`torch.backends.cuda.matmul.allow_fp16_accumulation=True`. Public input/output,
LayerNorm và residual vẫn FP32; thay đổi chỉ cho phép eligible internal FP16
GEMM accumulate FP16 thay vì FP32. Matrix/profile runner cô lập implementation
theo subprocess để flag không rò sang version khác.

Ý tưởng không qua gate. Với max-autotune và năm accuracy trial, #1–#9 và
#11–#13 PASS nhưng #10 FAIL `1/5,242,880` phần tử (`abs=0.00208624`, relative
khoảng `3.0%`). Với reduce-overhead, #8 FAIL `40,029/41,943,040` phần tử và
`max_abs=0.00657034`. Không row lỗi nào được benchmark.

Ngay trên #8 max-autotune đã PASS, paired measurement cũng không có gain: V4.3
`2.5258 ms`, V5.1 `2.5336 ms`, cùng 32 GPU events, 28 Triton events và cùng
kernel mix. Vì vừa mất correctness vừa không cải thiện đường max-autotune tốt
nhất, V5.1 chỉ được giữ làm negative ablation. Tên candidate MXFP8 cũ trong hồ
sơ được viết rõ là `V5.1-MXFP8`; alias `v5.1` trỏ tới ablation FP16 mới.

### 6.14 V6 — Tanh-approximated GELU ablation

`v6_ApproxGELU.py` kế thừa V4.3 và chỉ đổi optimized FFN activation từ
`F.gelu(..., approximate="none")` sang `approximate="tanh"`. Mixed-precision
cache, Linear, Flash-first SDPA, LayerNorm, residual, mask, state dict, public
FP32 output và safe fallback đều không đổi.

V6 PASS strict accuracy trên official #1–#13 với năm trial mỗi shape;
`failed=0` và max absolute error lớn nhất `0.00214118` ở #7. Các điểm vượt
absolute threshold vẫn PASS relative `<2%`. Local causal/non-causal ×
padding/no-padding diagnostics cũng PASS.

Không có measurable max-autotune gain. Ở clean paired #2, host median cho V4.3
`0.0780 ms`, V6 `0.0749 ms`, nhưng raw device time lại `0.0554 → 0.0560 ms`;
V6 còn tăng một compiled kernel do mất một epilogue fusion. Tín hiệu trái nhau
nên không đủ làm performance claim. Clean paired #8 cho `2.5502 → 2.5584 ms`,
tức V6 chậm hơn `0.32%`, với cùng kernel/CUDA Graph structure. Full matrix timing
không được dùng vì một ResNet training process khác chiếm `53–80%` GPU vật lý
#1; reverse-order retries cũng bị một evaluation workload mới làm nhiễu. V6
được giữ làm ablation nhưng V4.3 vẫn là best path.

### 6.15 V7 — Residual + LayerNorm pipeline ablation

`v7_ResidualLayerNorm.py` là V7a pure PyTorch. Attention/FFN projection giữ
output FP16 tới residual boundary; helper boundary cộng vào residual FP32, áp
mask đúng ordering, chạy LayerNorm FP32 rồi trả đồng thời residual FP32 và
normalized activation FP16 cho GEMM kế tiếp. Boundary cuối pipeline FFN
residual với final LayerNorm và vẫn zero invalid output sau norm. Exact GELU,
Flash-first SDPA, weights/cache/state dict và safe fallback giữ nguyên V4.3.

Local official #2 PASS 5/5 (`max_abs=0.00115311`); causal/non-causal ×
padding/no-padding cùng `valid_token_mask=None` diagnostics PASS và khớp V4.3
eager bit-for-bit. Training/BF16 fallback khớp exact, state-dict keys giữ
nguyên. GPU max-autotune #7/#10 PASS 5/5 với max abs lần lượt `0.00188218` và
`0.00140238`.

Paired profile không cho codegen mới. #2 có raw GPU time cùng `0.0546 ms`; #8
cho V4.3 `2.5155 ms`, V7a `2.5235 ms` (`-0.32%`). #12 nhìn như V7a nhanh hơn
trong thứ tự đầu nhưng effect gần biến mất khi đảo thứ tự, còn raw GPU time đổi
dấu. Mỗi cặp có cùng kernel count, Triton count và event names, gồm fused
`addmm + cast + residual + LayerNorm` cùng biến thể mask. Inductor đã thực hiện
fusion mục tiêu; V7a không được promote và V7b standalone Triton không được
viết vì có nguy cơ phá GEMM-template fusion hiện tại.

### 6.16 V8 — Shape-dispatched FFN-in GEMM + exact GELU

`v8_FusedFFNGELU.py` kế thừa V4.3 và dùng `torch.library.custom_op` để thay
`FFN-in Linear → exact GELU` bằng một Triton kernel ở workload phù hợp. Kernel
dot activation/weight FP16 với accumulator FP32, cộng bias, round Linear output
về FP16 giống V4.3, rồi tính exact erf-GELU từ giá trị đã round và store FP16
cho FFN-out. Attention, FFN-out, residual/LayerNorm FP32, cache/state dict và
training/non-FP32 fallback không đổi. Fake registration cho phép Inductor capture
custom op; máy không có Triton dùng exact PyTorch fallback.

Microbenchmark cho thấy tính phụ thuộc shape mạnh. #8 (`K=N=1024`) có isolated
gain `2.497x`, nhưng three-order whole-model measurements chỉ dao động quanh
zero dù kernel count giảm `32 → 29`. Ngược lại official #6, với 1.28 triệu
token rows và `D=FFN=128`, lặp lại win theo cả hai thứ tự: latency giảm `4.75%`
và `4.94%`. Lượt chốt accuracy-5 cho V4.3 `26.6799 ms`, V8 `25.4092 ms`
(`-4.76%`), raw GPU `26.6680 → 25.3948 ms`, kernels `32 → 29`.

Dispatcher vì vậy chỉ bật custom kernel khi `batch_size * seq_len >= 1_000_000`
và `d_model == ffn_dim == 128`; mọi shape khác chạy nguyên V4.3 `_mixed_ffn`.
Full max-autotune matrix official #1–#13 PASS 5/5 mỗi shape, max abs lớn nhất
`0.00188218`. Shape #14 vẫn pending.

### 6.17 V8.1 — Force-all FFN/GELU ablation

`v8_1_FusedFFNGELUAll.py` không thêm kernel mới: class chỉ kế thừa V8 và ép
`_use_fused_ffn_gelu=True` sau construction. Mục tiêu là đo trực tiếp việc bỏ
static dispatcher mà không sửa V8 stable.

V8.1 PASS strict accuracy 5/5 trên official #1–#13. Paired max-autotune theo
cả hai implementation orders cho geomean latency giảm `1.365%` và `0.969%` so
với V4.3. #1/#3/#4/#5/#10 giữ dấu win khoảng `1.2–2.7%`; #6 đạt
`26.6610 → 25.3857 ms` (`-4.78%`). Tuy nhiên #2 reverse-order regress `2.93%`,
#12 đổi từ `-2.52%` sang `+2.47%`, và #11 raw GPU time regress
`0.41–0.52%`. V8.1 vì vậy là dispatcher ablation, không phải unconditional
replacement; kernel count giảm không đủ làm promotion criterion.

### 6.18 V9 — Fully fused persistent MLP ablation

`v9_PersistentMLP.py` dùng một Triton custom op cho toàn bộ
`FFN-in → exact GELU → FFN-out`. Kernel tile token/output dimension, loop FFN
tiles, giữ hidden on-chip và chỉ store final projection. Cả hai dot accumulate
FP32; ba lần cast FP16 giữ rounding sau FFN-in+bias, GELU và FFN-out+bias như
V4.3. D/FFN multiples of 16 tới 128 dùng custom path; dimension lớn fallback.

Clean isolated sweep trên các unique supported official FFN shapes PASS và
nhanh hơn compiled two-GEMM reference `1.18–1.59x`. Full model #1–#13 cũng PASS
strict accuracy. Tuy nhiên whole-model #1/#5/#13 chậm hơn V4.3 lần lượt
`4.7%/2.0%/5.3%`; #2/#12 hòa. #7 giảm raw GPU time khoảng 12% nhưng primary
latency đổi dấu khi đảo implementation order. Trên #6, V9 `25.4481 ms` nhanh
hơn V4.3 `26.6497 ms` nhưng vẫn chậm hơn V8 `25.4071 ms`. V9 chỉ còn 25 kernels
so với V8 29 và V4.3 32–33, xác nhận launch count không thay thế end-to-end
measurement. Candidate được giữ làm ablation, không dispatch.

### 6.19 V11 arithmetic path — Exact GELU trực tiếp từ FP32 accumulator

`v11_FP32PreGELU.py` cô lập một precision-boundary ablation trên V8.1. V8 vốn
accumulate FFN-in dot product ở FP32 nhưng round `Linear+bias` xuống FP16 trước
khi exact GELU đọc lại giá trị đó. V11 bỏ riêng round-trip này: exact erf-GELU
tính trực tiếp từ FP32 accumulator cộng FP16 bias đã promote, rồi chỉ store
GELU output FP16 cho FFN-out. Activation/weight/bias cache FP16, attention,
FFN-out, residual/LayerNorm FP32, state dict và safe fallback giữ nguyên.

Kernel vẫn có cùng grid, autotune configs, một launch và một FP16 output tensor
như V8.1; vì vậy giả thuyết hiệu năng là latency trung tính. Custom op có fake
registration cho `torch.compile`; fallback CPU mô phỏng cùng precision boundary
bằng FP32 Linear từ các giá trị activation/weight/bias đã quantize FP16.

Paired local CPU diagnostic trên official #7/#10, PyTorch 2.12.1, 10 seed cho
mọi row strict PASS. Mean absolute error V11 tốt hơn V8.1 ở 10/10 trial của cả
hai shape: #7 giảm `0.00019114 → 0.00017636`, #10 giảm
`0.00017845 → 0.00016247`. Tuy nhiên metric sát comparator OR chưa Pareto-win:
worst normalized risk #7 đổi `0.802223 → 0.815028`, còn #10 cải thiện
`0.718771 → 0.654384`. Vì local fallback không chạy Triton, đây chỉ là
correctness-direction diagnostic.

Trên RTX 5090/PyTorch `2.13.0+cu130`, max-autotune V11 PASS strict official
#1–#13, failed `0`, worst max abs `0.00179082` ở #7. Paired hai implementation
orders so với V8.1 cho kết quả theo shape:

| Shape | Max-abs Δ | Host latency Δ | Raw GPU Δ | Kết luận |
|---:|---:|---:|---:|---|
| #7 | `-4.85%` | `+1.47% / +2.77%` | `+0.74% / +0.10%` | Accuracy tốt hơn, latency regress nhỏ |
| #10 | `-1.19%` | `-1.50% / -1.50%` | `-1.17% / -1.11%` | Pareto improvement |
| #6 | `-13.01%` | `+0.10% / +0.02%` | `+0.003% / -0.064%` | Accuracy tốt hơn, latency trung tính |

Kernel topology không đổi: 29 GPU kernels, 2 memory events, 21 Triton events và
một CUDA Graph launch/forward. D-027 promote force-all V11 làm main để ưu tiên
accuracy margin và một implementation thống nhất; regression #7 là trade-off
được chấp nhận, không phải speedup claim. Shape #14 vẫn pending.

### 6.20 V12 — FFN-out trực tiếp FP32

`v12_FP32FFNOut.py` kế thừa V11 và chỉ thay output precision của FFN-out. V11
dùng hidden/weight/bias FP16, accumulate GEMM FP32 nhưng store `F.linear` output
FP16 trước `.float()` và residual add. V12 dùng
`torch.mm(..., out_dtype=torch.float32)` trên CUDA rồi cộng cached FP16 bias đã
promote FP32, nên bỏ đúng lần round FP16 cuối FFN branch. CPU diagnostic promote
các operand đã quantize FP16 rồi chạy FP32 Linear để mô phỏng cùng boundary.

Thay đổi không nhằm bỏ kernel cast: compiled profile cũ đã chứng minh Inductor
fuse cast với residual/LayerNorm. FP32 output còn tăng intermediate traffic, nên
candidate chỉ được xem là accuracy/latency trade-off. Local strict
causal/non-causal × padding/no-padding, state dict/cache, training, FP16/BF16
fallback và `torch.compile` smoke đều PASS. Trên paired local 10 seed, mean
absolute error tốt hơn V11 ở 10/10 seed cho cả #7 và #10; worst max abs giảm
`0.00186644 → 0.00158411` ở #7 và `0.00174332 → 0.00156116` ở #10. Đây chưa
phải GPU evidence; CUDA strict gate và paired max-autotune còn pending.

Hai version phụ cô lập phần còn lại: V12.1 chỉ đổi attention out-projection;
V12.2 đổi cả attention-out lẫn FFN-out. Causal Flash-first, non-causal masked
fallback và mọi precision boundary khác vẫn giữ V11. Paired local 10 seed:

| Shape | V11 worst max abs | V12 FFN-only | V12.1 attention-only | V12.2 cả hai |
|---:|---:|---:|---:|---:|
| #7 | 0.00186644 | 0.00158411 | 0.00170401 | **0.00152946** |
| #10 | 0.00174332 | 0.00156116 | **0.00145376** | 0.00146723 |

V12.2 có mean absolute error thấp nhất ở cả hai shape, còn V12.1 nhỉnh hơn ở
worst max abs #10. Đây chỉ là CPU direction diagnostic; không cộng hoặc suy
performance từ các kết quả này.

### 6.21 V13 — INT8 FFN-in accuracy-only negative ablation

`v13_INT8FFNProbe.py` kiểm tra numerical feasibility trước khi đầu tư vào
Triton/CUTLASS INT8. Candidate kế thừa V11 và chỉ thay FFN-in: cached weight
dùng signed symmetric INT8 với per-output-channel scale; LayerNorm output vẫn
đi qua FP16 boundary của V11 rồi activation dùng dynamic symmetric per-token
scale. W8A8 dot dùng `torch._int_mm` với INT32 accumulator, dequantize FP32,
cộng cached FP16 bias đã promote và chạy exact GELU FP32; GELU output vẫn FP16
cho FFN-out. Attention, FFN-out, residual/LayerNorm, state dict và fallback
không đổi.

Probe có controls `TECHJAM_INT8_PROBE_MODE=w8|a8|w8a8` và layer selection qua
`TECHJAM_INT8_PROBE_LAYERS`. PyTorch quantize/dequantize graph chỉ dùng để đo
error; latency của nó không đại diện custom INT8 kernel.

Official shape #2 trên CPU/PyTorch 2.12.1, năm seed, fail toàn bộ: W8-only
`1,659/81,920` (`max_abs=0.00738502`), A8-only `4,574/81,920`
(`0.0122206`), W8A8 `5,825/81,920` (`0.0198889`). Scope nhỏ nhất cũng không
cứu được recipe: chỉ W8 FFN-in layer cuối vẫn fail `69/81,920`
(`0.00381088`), còn W8A8 layer cuối fail `1,332/81,920` (`0.00912209`). Vì
canary fail trước performance, project không chạy GPU #6/#8, không viết kernel
và không có INT8 speedup claim.

### 6.22 V14 — Memory-bounded batch chunking cho shape #14

Shape #14 có input và output FP32 `[32,100000,1024]`, mỗi tensor chiếm
`12.207 GiB`. V11 còn tạo packed QKV FP16 `[32,100000,3072]` cỡ
`18.311 GiB`; probe optimized-only đạt peak `30.588 GiB` rồi OOM ngay tại
allocation này. Flash Attention đã tránh score matrix nhưng không giải quyết
packed projection live set.

`v14_BatchChunked.py` dùng tính độc lập theo batch: preallocate đúng một output
full-size, chạy từng sample `B=1` qua đủ hai Transformer layers bằng nguyên V11,
rồi copy slice vào output. Không chunk giữa layer, không đổi QKV/attention/FFN,
weights, mask, dtype hay precision boundary. Dispatch chỉ bật khi config/input
khớp chính xác #14, FP32 eval, causal; mọi branch khác fallback V11. Local
causal + prefix-padding smoke cho output bitwise-identical V11; training và
BF16 branch không vào fast path.

Trên RTX 5090 physical index `1`, một `B=1/S=100000` probe peak `2.964 GiB`.
Full V14 forward sinh đúng FP32 output `[32,100000,1024]`, peak `28.526 GiB` và
first-call wall time `8.974 s`. Original baseline không thể làm accuracy trực
tiếp vì score `[32,16,100000,100000]` cần khoảng `18.6 TiB`, nên
`shape14_accuracy.py` giữ nguyên baseline formula nhưng chia Q thành block 256,
chạy một batch sample mỗi lần và apply nguyên strict comparator theo token
chunks. Reduced-shape equivalence với original baseline có max abs
`3.576e-7`, strict failed `0`.

Full seed-1234 validation PASS `0/3,276,800,000`, max abs `0.000831008`, mean
abs `6.56362e-05`, elapsed `327.725 s`, peak `21.133 GiB`. Sau accuracy gate,
optimized-only CUDA Event diagnostic một warmup/năm repeat đạt median
`6683.9873 ms`, mean `6682.2406 ms`, p90 `6700.7756 ms`, min `6656.8384 ms`,
throughput `478,756.15 token/s`, peak timed `27.000 GiB`. Baseline latency và
speedup là N/A vì reference không executable; số này không phải paired
performance claim. Repeated runner phải giải phóng output 12.21 GiB và collect
Python cycles giữa call ngoài timing, nếu không output trước có thể làm call kế
OOM. D-029 ban đầu giữ V14 ngoài `main.py`; quyết định đó được D-030 supersede
sau khi tạo V14.1 với compile-safe large-sequence helper.

### 6.23 V14.1 — Unified parent với cutoff theo `S`

`v14_1_BatchChunked.py` hợp nhất hai vai trò mà không trộn arithmetic: dưới
cutoff nó gọi thẳng V11; FP32 eval với `B > 1` và `S >= 8192` dùng cùng loop
batch-chunk size 1 đã validate ở V14. Official #1–#13 có `S <= 1024`, còn #14
có `S=100000`, nên dispatcher tách đúng hai workload groups bằng một guard rẻ.
Training, non-FP32 và `B=1` giữ nguyên V11 fallback.

Large-sequence helper dùng `torch.compiler.disable`. Nếu `torch.compile` capture
loop 32 sample của #14, graph có thể bị unroll và làm mất memory bound; graph
break có chủ đích giữ helper eager, trong khi nhánh V11 nhỏ vẫn compile bằng
Inductor. Local causal + prefix-padding diagnostic cho cả fallback và chunk
path bitwise-identical V11 (`max_abs=0`), state-dict keys không đổi và
training/BF16/B=1 fallback PASS. D-030 từng promote V14.1 qua `main.py`; D-031
sau đó giữ nó làm parent/rollback dưới V15. V11 và V14 vẫn là ablations.

### 6.24 V15 — Direct-layout QKV projection implementation

Compiled profile #13 cho thấy Inductor đã fuse residual/mask/LayerNorm/cast,
nên V15 không viết lại LayerNorm sau negative V7. Thay vào đó,
`v15_DirectQKVLayout.py` thay riêng packed QKV projection trên exact official
#13. Triton GEMM nhận FP16 activation/weight/bias, accumulate FP32 và map store
epilogue trực tiếp vào contiguous `[3,B,H,S,Dh]`. Flash Attention nhận ba
view contiguous `[B,H,S,Dh]` thay vì view interleaved có sequence stride `3D`.

Candidate kế thừa V14.1 và chỉ dispatch cho
`B64/S1024/D128/H4/L4/FFN128`, causal FP32 eval. Mọi shape/branch khác, kể
cả #14, giữ nguyên V14.1. CPU fallback materialize cùng direct layout bằng
PyTorch để test semantics; nó không phải performance path.

Local strict diagnostic ba seed, có/không prefix padding, PASS
`0/26,112`, max abs `0.00111527`; V14.1/V15 khớp bitwise, state dict/cache,
training/BF16 fallback và CPU Inductor smoke đều PASS. Remote PyTorch
`2.13.0+cu130`/Triton `3.7.1` import/`py_compile` PASS; Triton `ASTSource`
cross-compile cả sáu SM120 configs cũng PASS trước CUDA execution.

Trên RTX 5090 idle, custom projection khớp packed `F.linear` bitwise và official
#13 PASS năm trial `0/41,943,040`, max abs `0.00147235`. Paired max-autotune
dài (`50/200/5`) giữ gain ở cả hai orders: V14.1→V15 giảm median
`1.1080 → 1.0971 ms` (`-0.98%`) và raw GPU `1.0966 → 1.0838 ms`
(`-1.17%`); reverse-order, viết theo V14.1→V15, giảm `1.1251 → 1.1003 ms`
(`-2.20%`) và raw GPU `1.1062 → 1.0867 ms` (`-1.76%`). Triton GPU events giảm
`21 → 17`; Flash time gần trung tính, nên gain được quy cho QKV/projection-layout
region. Prefix-padding 25% exact #13, non-causal fallback và official #2/#12
fallback canaries đều PASS. D-031 promote V15 qua `main.py`; V14.1 là rollback.

#### 6.24.1 V15.1 — Cross-shape direct-layout QKV ablation

`v15_1_DirectQKVAll.py` dùng đúng V15 operator nhưng force flag cho mọi causal
config có `S<8192`; #14 vẫn fallback. Đây là one-variable measurement version:
không đổi weights, arithmetic boundary, `state_dict` hay public API. Local
branch/fallback gates PASS; official #1–#12 cũng PASS strict năm trial với tổng
`0/896,942,080` failed, max abs `0.00179085`.

Hai-order paired sweep cho thấy direct layout không phải global win. Chỉ #6
(`B=10000,S=128,D=FFN=128,H=4,L=4`) giảm đồng thời median end-to-end
`2.41–4.44%` và raw GPU `2.96–3.03%`; geometric ratios là `-3.43%/-2.99%`.
Ở workload này contiguous Q/K/V làm Flash nhanh hơn đủ để bù bốn direct-QKV
launches. Shape nhỏ không amortize được custom projection/layout; #2/#4 retest
dài còn xác nhận raw GPU regression `5.52–17.62%`.

D-035 vì vậy reject force-all V15.1 và không đổi main. #6 chỉ là dispatch
candidate; lần promote riêng phải dùng predicate workload large `B*S` thay vì
hardcode thêm exact official tuple, rồi rerun robustness/aggregate score.

### 6.25 V16 — Compile thân B=1, giữ loop #14 eager

`v16_CompiledBatchExecutor.py` kế thừa V15 nhưng override riêng helper
large-sequence. Vòng lặp 32 sample vẫn nằm dưới `torch.compiler.disable`, nên
Dynamo không unroll và peak memory vẫn bounded. Mỗi slice gọi cùng một compiled
bound method B=1; wrapper được tạo lazily sau weight load, device transfer và
`eval()`, rồi tái sử dụng. Compilation/autotune thuộc warmup, không nằm trong
CUDA Event timing.

Executor là derived cache lưu thẳng trong `__dict__`, không đăng ký child module
và không đổi `state_dict`. Cache bị invalidate sau `load_state_dict()`,
`_apply()`/`.to()` hoặc thay đổi train/eval mode. Nếu training, dtype khác FP32,
`S<8192` hoặc nhánh khác #14, V16 giữ nguyên V15/V14.1 behavior. Accuracy runner
gọi `forward_large_sequence_sample()` để kiểm tra đúng compiled graph B=1.

Local forced-chunk tests có/không mask khớp V15 bitwise; training, BF16,
short-sequence, cache lifecycle, Dynamo-eager và CPU Inductor đều PASS. CUDA
canary PyTorch 2.13 PASS; full gate trên RTX 5090/PyTorch `2.11.0+cu128` PASS
`0/3,276,800,000`, max abs `0.000944197`, mean abs `6.56367e-05`.

Exact-shape optimized-only sandwich V14.1→V16→V14.1, seed 1234, TF32,
warmup 1/repeats 5 cho medians `7396.7202 → 7166.8359 → 7435.5688 ms`.
V16 giảm latency `3.11–3.61%`, đạt `446,501 token/s`; timed peak giảm
`26.977 → 24.487 GiB`. Đây là head-to-head optimized comparison, không phải
speedup so với original baseline (baseline vẫn N/A). D-032 từng promote V16;
D-038 sau đó giữ nó làm direct-QKV rollback khi V16.1 trở thành main.

#### 6.25.1 V16.1 — Giữ compiled #14, bỏ direct-QKV #13

`v16_1_NoDirectQKV13.py` là source-clean main kế thừa trực tiếp V14.1 và
chứa riêng outer eager loop, reusable compiled B=1 executor cùng cache
invalidation đã validate ở V16. Nó không import/inherit V15/V16, không có
direct-QKV flag hoặc exact official tuple. MRO dừng ở
V16.1→V14.1→V11→V8→V4.3→mixed→baseline; parameter/state/public API không đổi.

Local #13 và forced large-sequence/prefix-mask outputs khớp V14.1 bitwise.
RTX 5090 official #13 PASS năm trial `0/41,943,040`, max abs `0.00147235`;
paired graph có cùng 29 kernels/21 Triton events và host delta đổi dấu
`+0.93%/-0.98%`. Exact-config #14 compiled B=1 canary PASS `0/102,400,000`,
max abs `0.000719786`. D-038 promote V16.1 làm main để bỏ exact-#13 tuple/V15
dependency. Đây là source-topology choice, không phải performance win: main mới
chủ đích trả lại gain `0.98–2.20%` của V15/V16 ở official #13.

#### 6.25.2 Standalone `v16_1_clean.py`

D-042 flatten toàn bộ MRO V16.1 vào một module: config và cấu trúc reference,
FP16 packed/projection cache, Flash-first causal SDPA, Triton FFN-in với exact
GELU đọc FP32 accumulator, reference fallback, large-sequence cutoff và reusable
compiled B=1 executor. Module chỉ phụ thuộc PyTorch và optional Triton; benchmark
CLI vẫn nằm riêng ở `main.py`.

Strict state dict, causal/non-causal × mask/no-mask, training/BF16 fallback và
forced large-sequence compiled-eager executor đều khớp composed V16.1 cũ
bit-for-bit trên local CPU. Tại thời điểm D-042 đây chỉ là packaging
equivalence; D-043 sau đó đóng full official GPU gate trên đúng file clean.

### 6.26 V17 — Compiled executor batch chunk 2

`v17_CompiledBatch2.py` là one-variable ablation trên V16: large-sequence
batch chunk tăng từ 1 lên 2, nên official #14 dùng 16 compiled executor calls
thay vì 32. Outer loop vẫn compiler-disabled; cache/invalidation, cutoff,
weights, mixed-precision boundaries và output contract không đổi. Vì B=2 sẽ
recursively vào V14.1 scheduler nếu gọi `super().forward`, inner callable gọi
thẳng V11 forward body—chính arithmetic mà V14.1 chỉ làm nhiệm vụ schedule.
Executor vẫn chấp nhận B=1 để xử lý batch lẻ.

Accuracy harness chạy candidate theo group B=2 rồi compare từng sample với
query-blocked reference, bảo đảm test đúng compiled graph mới. Local B2/B3
mask/no-mask và fallback branches khớp V16 bitwise. RTX 5090/PyTorch
`2.11.0+cu128` full #14 PASS `0/3,276,800,000`, max abs `0.000944197`, mean abs
`6.56366e-05`; #2/#13 fallback canaries cũng PASS.

Alternating V16→V17→V16→V17 với warmup 1/repeats 5 cho medians
`7183.8022 → 7141.3345 → 7162.9731 → 7131.5425 ms`. V17 nhanh hơn adjacent
controls `0.30–0.59%`; trung bình hai medians giảm `0.515%`, nhưng timed peak
không đổi `24.487 GiB` và từng process có drift theo repeat. D-033 giữ V17 làm
ablation thay vì thêm specialization B=2 vào main; V16 vẫn promoted ở thời điểm
đó, trước khi D-038 chọn V16.1 vì source topology.

### 6.27 V18 — SageAttention exact-#14 negative ablation

Profiler mới cho inner executor V16 B=1 ghi nhận attention chiếm `92.258%`
raw device time, nên V18 thử thay đúng attention core của official #14 bằng
SageAttention INT8-QK per-thread, PV-FP16 với FP32 accumulation. QKV projection,
output projection, FFN, FP32 residual/LayerNorm, cutoff và batch chunk B=1 giữ
nguyên V16. Optional import bảo đảm host không có Sage và mọi non-target branch
fallback về V16.

Isolated exact attention cho thấy upside `100.5045 → 72.1337 ms` (`1.3933x`)
nhưng đã fail strict `94/102,400,000`. Full-model eager B=1 attenuate gần hết
sai số nhưng vẫn **FAIL `1/102,400,000`**, max abs `0.0026415`. Vì comparator
dùng strict OR, một outlier vẫn loại candidate trước performance. Custom-op
compiled integration còn không equivalent eager (`59,375,874/102,400,000`
fail); fake stride contract đã được sửa nhưng semantic wrapper vẫn chưa đạt
gate. V18 không được promote, không có model timing và không thêm SageAttention
vào dependency ổn định. D-034 giữ V16/PyTorch Flash làm main.

### 6.28 V17-Sage — SageAttention với exact-prefix correction

`v17_sage.py` là candidate mới theo tên owner yêu cầu; nó không ghi đè
historical `v17_CompiledBatch2.py`. Class kế thừa source-clean V16.1, dùng Sage
INT8-QK per-thread + PV-FP16/FP32-accum cho causal FP32 eval khi `S>32` và
`head_dim<=128`. Sage 2.2 tự pad head dimension nhỏ lên 64/128; official #8 có
`head_dim=256` và #12 có `S=32` nên fallback nguyên V16.1. CPU, non-causal,
training, public dtype khác FP32 hoặc thiếu optional dependency cũng fallback.

Accuracy correction có evidence trực tiếp từ
`shape14_sage_locality_seed1234_20260830.json`: 109/102.4M isolated violations
đều nằm ở query `1..31`, `minimal_exact_prefix=32`. Candidate chạy Sage toàn
attention, chạy PyTorch Flash chính xác trên causal square prefix 32 rồi
overwrite prefix output. Vì 32 query đầu không thể nhìn key từ vị trí 32 trở
đi, phép tính prefix là đúng semantics và chỉ tốn `O(32^2)` attention work.
Attention out-projection còn store trực tiếp FP32 accumulator theo helper V12.1
để giảm thêm rounding trước residual.

V18 compiled failure được xử lý ở integration layer: custom op được tag
`torch.Tag.cudagraph_unsafe`, fake output khai báo contiguous HND và compiled
large-sequence default là `max-autotune-no-cudagraphs`. Hai env controls phục vụ
audit: `TECHJAM_SAGE_REQUIRE=1` cấm silent dependency fallback, còn
`TECHJAM_SAGE_EXACT_PREFIX` đổi prefix cho accuracy ablation.

Full RTX 5090/PyTorch `2.11.0+cu128` official #1–#13 matrix sau đó reject
candidate: #6 fail strict với max abs `0.00250164`, #9 fail `0.00255397`.
#13 PASS và đạt baseline/optimized `41.9384/2.4573 ms` (`17.067x`), nhưng
correctness là gate toàn cục. #8 và #12 là V16.1 fallback, không phải Sage win.
V17-Sage giữ làm negative ablation; V16.1 vẫn là main theo D-040.

### 6.29 V18-Sage — direct automatic performance-only probe

`v18_sage.py` kế thừa trực tiếp source-clean V16.1, không kế thừa V17-Sage hay
historical `v18_SageAttentionShape14.py`. Candidate thay đúng SDPA attention
core bằng `sageattention.sageattn` automatic. Ở SM120/source commit pin,
dispatcher dùng INT8-QK per-warp và FP8-PV `fp32+fp16`.

V18-Sage chủ đích bỏ exact-prefix repair, prefix copy và FP32 attention
out-projection của V17; QKV/out projection, V11 FFN, residual/LayerNorm,
state-dict và V16.1 large-sequence schedule giữ nguyên. Sage bật trên causal
FP32 CUDA eval với original `head_dim<=128`, kể cả #12 `S=32`. Official #8
`head_dim=256` cùng CPU/training/non-causal/non-FP32 fallback V16.1.

Custom op `techjam::sage_attention_auto_v18` khai báo contiguous output và
`cudagraph_unsafe`; shape-#14 executor dùng
`max-autotune-no-cudagraphs`. `TECHJAM_SAGE_REQUIRE=1` khóa dependency/version
để không silent fallback. `matrix_runner --benchmark-on-failure` chỉ phục vụ
performance diagnostic: strict comparator/tolerance vẫn chạy, row fail vẫn là
`ACCURACY_FAIL` và timing không hợp lệ để promote hay claim official score.

### 6.30 V19 — CUDA FP16 accumulation với checkpoint FP32

V19 thay đúng kernel `FFN-in -> exact GELU` của V16.1. CUDA extension stage
activation và weight vào shared memory theo block output `64x32`; tám warps chạy
WMMA `16x16x16`. Mode mặc định `K=32` dùng accumulator FP16 trong hai MMA liên
tiếp, store partial tile rồi promote thủ công vào register FP32 trước khi reset
accumulator FP16. Sau khi quét hết K, bias và exact erf-GELU chạy từ tổng FP32;
GELU output vẫn store FP16 cho FFN-out như V16.1.

Thiết kế không bật `torch.backends.cuda.matmul.allow_fp16_accumulation`, nên
không có process-global side effect như V5.1. Env
`TECHJAM_V19_CHECKPOINT_K=16|32|64|128|fp32` cung cấp bốn khoảng checkpoint và
một WMMA-FP32 control trên cùng block/layout/epilogue. Extension được build khi
model chuyển sang CUDA, trước `torch.compile`; build failure mặc định dừng run
để không silent đo fallback. `main.py` vẫn là V16.1.

Local CPU/PyTorch 2.12.1 đã PASS strict state dict, causal/non-causal x
mask/no-mask, training/BF16/unsupported-shape fallback, custom-op `opcheck` và
`torch.compile(backend="eager")`. Official shape #2 một-trial portable smoke
PASS cả K=16/32/64/128 và FP32 control; default K=32 có max abs `0.00108075`,
failed `0/16,384`. Portable CPU simulation không tái tạo chính xác tensor-core
FP16 accumulation, nên kết quả này chỉ là structural gate.

GPU RTX 5090/driver 595 đã đóng gate. NVCC cần literal FP16 đúng type trong
`wmma::fill_fragment`; sau fix, extension build/chạy thật và mọi K strict PASS
canaries. K64 là FP16 mode nhanh nhất trên #6 (`29.7675 ms`) nhưng V16.1 controls
chỉ `25.1593/25.2380 ms`. Full K64 #1–#13 PASS 13/13, worst max abs
`0.00181192`, geomean speedup `10.3079x` so với V16.1 cùng host `11.8030x`.
Full #14 PASS `0/3.2768B`, nhưng two-order medians `7251.4170/7310.5811 ms`.
V19 vì vậy bị reject cho promotion.

### 6.31 V19.1 — Parallel batch partitions

V19.1 thử song song hóa outer batch loop của large-sequence path thay vì đổi
executor B=1 bên trong. V19.1.0 kế thừa V16.1; V19.1.1 kế thừa V19. Cả hai dùng
chung `v19_parallel_batch_common.py`, nên comparison với parent cô lập đúng chi
phí/lợi ích của scheduler.

Env `TECHJAM_V19_PARALLEL_PARTS=1|2|4|8|16|32` chọn số partition, mặc định 2.
Các partition là range liên tục cân bằng; mỗi CUDA stream enqueue tuần tự các
sample B=1 thuộc range của nó. Scheduler thiết lập dependency current→worker
trước khi chạy và worker→current trước khi return, đồng thời record lifetime
của input/mask/output. Parts=1 và mọi branch không eligible gọi nguyên parent.

Concurrent calls không dùng CUDA Graph: parts>1 ép inner executor sang
`max-autotune-no-cudagraphs`, vì replay đồng thời một graph/static buffer cache
có thể race. Stream cache không persistent và public state dict không đổi.
Local planner/state/fallback gates cùng official #2 one-trial smoke PASS. GPU
sweep P2/P4/P8 đều exercise multi-stream path và strict PASS canaries; P16
không chạy vì P8 đã regress và resident memory khoảng `29.6 GiB`.

V19.1.0 chọn P4: full #14 strict PASS `0/3,276,800,000`, max abs
`0.000944138`; post-gate median `6780.3867 ms`, p90 `6792.4046 ms`, throughput
`471,949.48 token/s`, peak `25.676 GiB`. Hai control sandwiches đo gain
`1.51–1.66%` so với V16.1 P1 cùng no-CUDA-Graph policy. V19.1.1 K64/P2 cũng
full PASS nhưng two-order average `7179.4322 ms`, chậm hơn V19.1.0. `main.py`
chưa đổi; raw evidence nằm trong
`results/v19-tune-rtx5090-driver595-20260901/`.

## 7. Correctness validation

Accuracy dùng luật elementwise:

```text
abs(user - reference) < 0.002
OR
abs(user - reference) < 0.02 * abs(reference)
```

Max relative error có thể rất lớn tại reference values gần 0 nhưng case vẫn đúng nếu absolute error nhỏ hơn `0.002`. Vì vậy số phần tử fail mới là điều kiện quyết định, không phải chỉ nhìn `max_rel`.

### 7.1 Local matrix

Môi trường local: PyTorch `2.12.1`, CPU.

| Implementation | Dtype | Causal | Padding | Kết quả |
|---|---|:---:|:---:|---|
| V2 | FP32 | Tắt/bật | Không/có | PASS |
| V2 | BF16 | Bật | Có | PASS exact, reference-math fallback |
| V3 | FP32 | Tắt/bật | Không/có | PASS |
| V3 | BF16 | Bật | Có | PASS exact, full-model fallback |
| V3.1 | FP32 | Tắt/bật | Không/có | PASS |
| V3.1 | BF16 | Bật | Có | PASS exact, full-model fallback |
| V11 | FP32 public/FP16 internal | Tắt/bật | Không/có | PASS local diagnostic và GPU #1–#13 |
| V11 | FP16/BF16 public fallback | Bật | Có | PASS exact |
| V14 | FP32 public, exact #14 dispatch | Bật | All-valid | PASS GPU `0/3.2768B` |
| V17-Sage | FP32 public, corrected Sage | Bật | Có | GPU #1–#13 **FAIL** #6/#9; rejected |
| V18-Sage | FP32 public, dependency-missing CPU fallback | Bật | Có | Local fallback bitwise PASS; direct Sage GPU pending |
| V19 CUDA checkpoint | FP32 public / FP16 internal, K=16/32/64/128/fp32 | Tắt/bật | Không/có | GPU #1–#14 PASS; performance regress, rejected |
| V19.1.0/V19.1.1 | FP32 public, parallel outer batch scheduler | Bật | Không/có | GPU full #14 PASS; V19.1.0 P4 measured winner |
| V14.1 | FP32 public, `S >= 8192` dispatch | Bật | Không/có | Local branch equivalence PASS; #14 dùng cùng validated arithmetic |
| V15 | FP32 public, exact #13 direct QKV | Bật/tắt fallback | Không/có | Local/CUDA branch gates PASS; non-target path kế thừa V14.1 |
| V15.1 | FP32 public, causal `S<8192` direct QKV | Bật/tắt fallback | Không/có | Local gates và GPU official #1–#12 PASS; ablation, không phải main |
| V16.1 | FP32 public, V14.1 packed QKV + standalone #14 executor | Bật | Không/có | Source/MRO audit, local gates, GPU #13 và #14 B=1 canary PASS |

### 7.2 GPU matrix đã chạy

Môi trường: RTX 5090 vật lý index `1`, PyTorch `2.13.0+cu130`, CUDA `13.0`.

| Impl | Dtype/mask | Shape | Trials | Max abs | Failed elements | Kết quả |
|---|---|---|---:|---:|---:|---|
| V2 | FP32, non-causal, không padding | B8/S128/D512/H8/FFN2048/L6 | 3 | 0.000665478 | 0/1,572,864 | PASS |
| V2 | FP32, causal, không padding | B64/S128/D128/H4/FFN128/L4 | 3 | 0.00105309 | 0/3,145,728 | PASS |
| V3 | FP32, non-causal, không padding | B8/S128/D512/H8/FFN2048/L6 | 3 | 0.000665478 | 0/1,572,864 | PASS |
| V3 | FP32, causal, không padding | B64/S128/D128/H4/FFN128/L4 | 3 | 0.00105309 | 0/3,145,728 | PASS |
| V3 | FP32, causal + padding | B64/S128/D128/H4/FFN128/L4 | 3 | 0.00105309 | 0/3,145,728 | PASS |
| V3 | FP32, non-causal + padding | B8/S128/D512/H8/FFN2048/L6 | 3 | 0.000714183 | 0/1,572,864 | PASS |
| V3 | FP16, causal + padding | B8/S128/D128/H4/FFN128/L4 | 2 | 0 | 0/262,144 | PASS, fallback |
| V3 | BF16, causal + padding | B8/S128/D128/H4/FFN128/L4 | 2 | 0 | 0/262,144 | PASS, fallback |
| V3.1 | FP32, causal + padding | B64/S128/D128/H4/FFN128/L4 | 5 | 0.00105309 | 0/5,242,880 | PASS |
| V3.1 | FP32, causal, official #1 | B64/S128/D128/H4/FFN128/L4 | 5 | 0.00105309 | 0/5,242,880 | PASS |
| V3.1 | FP32, non-causal + padding | B8/S128/D128/H4/FFN128/L4 | 5 | 0.000709176 | 0/655,360 | PASS |
| V3.1 | FP16/BF16, causal + padding | B8/S128/D128/H4/FFN128/L4 | 2 mỗi dtype | 0 | 0/262,144 mỗi dtype | PASS, fallback |
| V3.1 | FP32, causal, official #13 | B64/S1024/D128/H4/FFN128/L4 | 5 | 0.00105309 | 0/41,943,040 | PASS |
| V4 FP16 | FP32 public / FP16 internal, official #1 | B64/S128/D128/H4/FFN128/L4 | 5 | 0.00159800 | 0/5,242,880 | PASS |
| V4.1 FP16 GELU | FP32 public / FP16 internal, official #1 | B64/S128/D128/H4/FFN128/L4 | 5 | 0.00159800 | 0/5,242,880 | PASS |
| V4.1 FP16 GELU | FP32 public / FP16 internal, official #2/#7/#8/#12/#13 | official dimensions | 5 mỗi shape | 0.00188218 tối đa | 0 | PASS selected matrix |
| V4.2 SDPA dispatch | FP32 public / FP16 internal, official #1–#13 | official dimensions | 5 mỗi shape | 0.00188218 tối đa | 0 | PASS partial matrix |
| V4.3 Flash-first | FP32 public / FP16 internal, official #1–#13 | official dimensions | 3 mỗi shape | 0.00188218 tối đa | 0 | PASS partial matrix |
| V4.3 causal Flash | Causal + right padding 25%/75%, diagnostic #1 | B64/S128/D128/H4/FFN128/L4 | 5 / 3 | 0.00159800 | 0 | PASS |
| V4.2 cuDNN | Causal/non-causal + 25% padding diagnostic | B64/S128/D128/H4/FFN128/L4 | 5 mỗi nhánh | 0.00159800 tối đa | 0 | PASS |
| V4 BF16 | FP32 public / BF16 internal, official #1 | B64/S128/D128/H4/FFN128/L4 | 5 | — | >0 | **FAIL** |
| V5 per-tensor FP8 | FP32 public / FP8 Linear / FP16 SDPA, official #2 | B1/S128/D128/H4/FFN128/L4 | 3 | 0.110912 | 25,429/49,152 | **FAIL** |
| V5.1-MXFP8 full MXFP8 | FP32 public / MXFP8 Linear / FP16 SDPA, official #2 | B1/S128/D128/H4/FFN128/L4 | 3 | 0.144848 | 25,800/49,152 | **FAIL** |
| V5.2 MXFP8 QKV-only | Phần còn lại V4.3 FP16, official #2 | B1/S128/D128/H4/FFN128/L4 | 3 | 0.066844 | 7,532/49,152 | **FAIL** |
| V5.1 FP16 accumulation, max-autotune | FP32 public / FP16 internal, official #10 | B64/S128/D128/H2/FFN128/L4 | 5 | 0.00223458 | 1/5,242,880 | **FAIL** |
| V5.1 FP16 accumulation, reduce-overhead | FP32 public / FP16 internal, official #8 | B64/S128/D1024/H4/FFN1024/L4 | 5 | 0.00657034 | 40,029/41,943,040 | **FAIL** |
| V6 approximate GELU | FP32 public / FP16 internal, official #1–#13 | official dimensions | 5 mỗi shape | 0.00214118 tối đa | 0 | PASS partial matrix |
| V7 residual/LayerNorm pipeline | FP32 public / FP16 internal, official #7/#10 | official dimensions | 5 mỗi shape | 0.00188218 tối đa | 0 | PASS targeted gate |
| V8 fused FFN/GELU dispatcher | FP32 public / FP16 internal, official #1–#13 | official dimensions | 5 mỗi shape | 0.00188218 tối đa | 0 | PASS partial matrix |
| V8.1 force-all FFN/GELU | FP32 public / FP16 internal, official #1–#13 | official dimensions | 5 mỗi shape | 0.00188218 tối đa | 0 | PASS partial matrix |
| V9 persistent full MLP | FP32 public / FP16 internal, official #1–#13 | official dimensions | 5 mỗi shape | 0.00188218 tối đa | 0 | PASS partial matrix |
| V11 FP32 pre-GELU | FP32 public / FP16 internal, official #1–#13 | official dimensions | 5 mỗi shape | 0.00179082 tối đa | 0 | PASS partial matrix |
| V14 batch-chunk | FP32 public / FP16 internal, official #14 | B32/S100000/D1024/H16/FFN1024/L2 | 1 full seed | 0.000831008 | 0/3,276,800,000 | **PASS** |
| V15 direct-layout QKV | FP32 public / FP16 internal, official #13 | B64/S1024/D128/H4/FFN128/L4 | 5 | 0.00147235 | 0/41,943,040 | **PASS** |
| V15 direct-layout QKV | Causal + 25% prefix padding, official #13 | B64/S1024/D128/H4/FFN128/L4 | 3 | 0.00147235 | 0/25,165,824 | **PASS** |
| V15.1 direct-QKV cross-shape | FP32 public / FP16 internal, official #1–#12 | official dimensions | 5 mỗi shape | 0.00179085 tối đa | 0/896,942,080 | **PASS** |
| V16 compiled B=1 executor | FP32 public / FP16 internal, official #14 | B32/S100000/D1024/H16/FFN1024/L2 | 1 full seed | 0.000944197 | 0/3,276,800,000 | **PASS** |
| V16.1 no-direct-QKV13 | FP32 public / FP16 internal, official #13 | B64/S1024/D128/H4/FFN128/L4 | 5 | 0.00147235 | 0/41,943,040 | **PASS** |
| V16.1 compiled executor canary | FP32 public / FP16 internal, official #14 config | B1/S100000/D1024/H16/FFN1024/L2 | 1 batch canary | 0.000719786 | 0/102,400,000 | **PASS** |
| V17 compiled B=2 executor | FP32 public / FP16 internal, official #14 | B32/S100000/D1024/H16/FFN1024/L2 | 1 full seed | 0.000944197 | 0/3,276,800,000 | **PASS** |
| V18 Sage PV-FP16 eager canary | FP32 public / FP16 internal, official #14 config | B1/S100000/D1024/H16/FFN1024/L2 | 1 batch canary | 0.0026415 | 1/102,400,000 | **FAIL** |
| V17-Sage corrected | FP32 public / Sage supported shapes, official #1–#13 | official dimensions | 5 mỗi shape | 0.00255397 tối đa | #6/#9 >0 | **FAIL** |
| V13 INT8 FFN-in probe | W8/A8/W8A8 fake quant, official #2 | B1/S128/D128/H4/FFN128/L4 | 5 mỗi mode | 0.0198889 tối đa | >0 mọi mode | **FAIL** |
| V16.1 standalone final | FP32 public / FP16 internal, official #1–#13 | official dimensions | 5 mỗi shape | 0.00179085 tối đa | 0 | **PASS** |
| V16.1 standalone final | FP32 public / FP16 internal, official #14 | B32/S100000/D1024/H16/FFN1024/L2 | 1 full seed | 0.000944197 | 0/3,276,800,000 | **PASS** |

Promoted V16.1 main thống nhất V14.1/V11 packed-QKV path cho #1–#13 với
standalone compiled executor #14, không chứa exact official tuple. V16 giữ làm
rollback nếu cần direct-QKV #13. Fresh run ngày 2026-08-31 đã đóng full gate
trên đúng standalone artifact: #1–#13 PASS năm trial, còn #14 PASS đủ 32 batch.
Multi-seed/input-scale/padding vẫn là robustness work; paired performance
baseline #14 vẫn không có. V8 đã PASS năm trial trên #1–#13, gồm shape #6
batch `10000`.

## 8. GPU benchmark

### 8.1 Môi trường

| Thành phần | Giá trị |
|---|---|
| Provider / OS | Vast.ai container, Ubuntu 24.04.4, kernel `5.15.0-187-generic` |
| CPU | AMD Ryzen 5 5600X 6-Core Processor; 12 logical CPUs visible |
| RAM / workspace disk | `33,564,246,016` bytes / 16 GiB overlay |
| GPU | NVIDIA GeForce RTX 5090, compute capability `12.0`, 32,607 MiB VRAM |
| Driver | `595.71.05` |
| Python | `3.12.14` |
| PyTorch / CUDA wheel | `2.11.0+cu128` / `12.8` |
| cuDNN / Triton | `9.19.0` / `3.6.0` |
| Dtype | FP32 |
| TF32 | Bật cho baseline và optimized |
| Seed | `1234` |
| Warmup/repeats/rounds | `20 / 100 / 3` |
| Git revision | `4f77a04fd51c3a2ad8b9d8986657915ae3ca94d6` |

Final instance chỉ có một GPU nên dùng `CUDA_VISIBLE_DEVICES=0`; PyTorch gọi nó
là `cuda:0`. GPU idle trước run, visible memory usage `2 MiB`, utilization `0%`.
Machine-readable inventory và raw result nằm trong `results/final/`.

Các số lịch sử ở §8.2–§8.19 còn đến từ máy phát triển Debian/PyTorch
`2.13.0+cu130` hoặc một Vast.ai stack trước đó; mỗi subsection giữ environment
và command riêng. Không trộn chúng vào final active-main aggregate.

Baseline và optimized final #1–#13 được đo nối tiếp trong cùng subprocess với
thứ tự đảo theo round. Shape #14 dùng harness riêng vì original reference không
executable trên 32 GiB.

### 8.2 Kết quả official shape #1

| Config | Candidate | Baseline median | Candidate median | Speedup |
|---|---|---:|---:|---:|
| Official shape #1 causal | V2 | 1.0120 ms | 0.7469 ms | 1.355x |
| Official shape #1 causal | V3 | 1.0558 ms | 0.5494 ms | **1.922x** |
| Official shape #1 causal | V3.1 | 1.0021 ms | 0.4656 ms | **2.152x** |

P90 không được giữ trong log lịch sử của v2. Với lượt v3 official shape #1:

| Config | Baseline median / p90 | V3 median / p90 | V3 throughput | Speedup |
|---|---:|---:|---:|---:|
| B64/S128/D128/H4/FFN128/L4, causal | 1.0558 / 1.1387 ms | 0.5494 / 0.5971 ms | 14,911,463 token/s | **1.922x** |

So với v3 intermediate có no-copy views nhưng chưa flatten, official causal latency giảm từ `0.6874` xuống `0.5494 ms`, tương đương `20.1%`.

### 8.3 V3 so với v3.1 trên cùng official shapes

Hai implementation được chạy bằng matrix runner trong hai subprocess riêng nhưng cùng GPU, dtype, seed, TF32, accuracy trials `5`, warmup `20`, repeats `100` và rounds `3`.

| Shape | Impl | Baseline median / p90 | Optimized median / p90 | Speedup |
|---|---|---:|---:|---:|
| #1, B64/S128/D128/H4/L4/FFN128 | V3 | 1.0252 / 1.0370 ms | 0.5322 / 0.5436 ms | 1.926x |
| #1, B64/S128/D128/H4/L4/FFN128 | V3.1 | 1.0021 / 1.0168 ms | 0.4656 / 0.4777 ms | **2.152x** |
| #13, B64/S1024/D128/H4/L4/FFN128 | V3 | 41.7561 / 41.7769 ms | 9.7829 / 9.8054 ms | 4.268x |
| #13, B64/S1024/D128/H4/L4/FFN128 | V3.1 | 41.7584 / 41.7826 ms | 5.4255 / 5.4438 ms | **7.697x** |

V3.1 giảm optimized median `12.5%` ở shape #1 và `44.5%` ở shape #13 so với v3.

### 8.4 V3.1 eager so với `torch.compile` trên official shape #1

Hai process dùng cùng GPU/dtype/shape/seed/TF32, accuracy trials `3`, warmup `50`, repeats `200`, rounds `5`; chỉ compiled process thêm `--compile-user --compile-mode reduce-overhead`.

| Execution | Accuracy | Baseline median / p90 | V3.1 median / p90 | Throughput | Speedup |
|---|---|---:|---:|---:|---:|
| Eager | PASS, max_abs=0.00105309 | 1.0806 / 1.1374 ms | 0.5118 / 0.5371 ms | 16,007,004 token/s | 2.112x |
| Compile `reduce-overhead` | PASS, max_abs=0.00105298 | 1.0786 / 1.1175 ms | **0.3169 / 0.3190 ms** | 25,853,362 token/s | **3.404x** |

Compiled v3.1 giảm median `38.1%` và tăng throughput `61.5%` so với eager v3.1. Đây là steady-state result; compile/capture time không nằm trong timed forward. `default`/`max-autotune`, compiled kernel trace và full 14-shape matrix vẫn pending nên chưa kết luận phần gain riêng của fusion hay CUDA Graph.

### 8.5 V4/V4.1 mixed precision trên official shape #1

Paired profiler run dùng accuracy trials `5`, warmup `50`, repeats `200`, rounds
`5` trên cùng RTX 5090/PyTorch `2.13.0+cu130`:

| Execution | Candidate | Accuracy | Candidate median / p90 | Speedup so với baseline cùng process |
|---|---|---|---:|---:|
| Eager | V3.1 | PASS | 0.5136 / 0.5280 ms | 2.115x |
| Eager | V4 FP16 | PASS | 0.5836 / 0.6433 ms | 1.764x |
| Eager | V4.1 FP16 GELU | PASS | 0.5492 / 0.5880 ms | 1.864x |
| Compile `reduce-overhead` | V3.1 | PASS | 0.3168 / 0.3189 ms | 3.247x |
| Compile `reduce-overhead` | V4 FP16 | PASS | **0.1858 / 0.1860 ms** | 5.861x |
| Compile `reduce-overhead` | V4.1 FP16 GELU | PASS | **0.1858 / 0.1860 ms** | 5.828x |

Speedup rows dùng baseline median riêng của từng subprocess, vì baseline dao động
`1.0235–1.0889 ms`; so candidate median trực tiếp cho thấy V4/V4.1 compiled nhanh
hơn V3.1 compiled `41.4%`. Manual FP16 GELU chỉ cải thiện eager vì Inductor đã
fuse conversion trong cả hai compiled graph.

### 8.6 V4.2 SDPA dispatcher trên official shapes #1–#13

RTX 5090, PyTorch `2.13.0+cu130`, FP32 public/FP16 internal,
`torch.compile(mode="reduce-overhead")`, accuracy trials `5`, warmup `20`,
repeats `100`, rounds `3`; mọi row `failed=0`:

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

Geomean speedup #1–#13 là `7.58x`, tăng `6.9%` so với khoảng `7.09x` của V4.1
matrix. Đây là partial official matrix; shape #14 không bị loại khỏi yêu cầu cuối.

### 8.7 V4.3 Flash-first trên official shapes #1–#13

RTX 5090, PyTorch `2.13.0+cu130`, FP32 public/FP16 internal,
`torch.compile(mode="reduce-overhead")`, accuracy trials `3`, warmup `20`,
repeats `100`, rounds `3`; mọi row `failed=0`:

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

Geomean #1–#13 là `8.52x`. Profiler #13 xác nhận
`pytorch_flash::flash_fwd_kernel`, 33 GPU kernels, một compiled region và một
CUDA Graph launch; attention device time khoảng `0.4211 ms`. CPU causal
diagnostic PASS qua Math fallback; non-causal + padding PASS masked automatic
path. Shape #14 vẫn pending.

### 8.8 V4.3 max-autotune trên official shapes #1–#13

Cùng RTX 5090/PyTorch `2.13.0+cu130`, FP32 public/FP16 internal, warmup `20`,
repeats `100`, rounds `3`; mọi row PASS strict accuracy:

| ID | V4.3 max-autotune median | Speedup |
|---:|---:|---:|
| 1 | 0.1379 ms | 7.476x |
| 2 | 0.0760 ms | 13.769x |
| 3 | 0.0801 ms | 14.791x |
| 4 | 0.0750 ms | 14.077x |
| 5 | 0.2124 ms | 5.158x |
| 6 | 26.6779 ms | 6.631x |
| 7 | 0.0730 ms | 14.317x |
| 8 | 2.5400 ms | 2.472x |
| 9 | 0.1510 ms | 6.321x |
| 10 | 0.1366 ms | 7.464x |
| 11 | 0.1758 ms | 9.032x |
| 12 | 0.0812 ms | 12.789x |
| 13 | 1.1096 ms | 37.649x |

Geomean speedup là `9.5266x`, tăng `11.83%` so với matrix reduce-overhead
`8.5186x`; geomean optimized latency giảm `9.73%`. Đây là compile steady-state,
không gồm autotune/compile time. Artifact:
`benchmark-results/matrix_v4_3_Flash_float32_20260828T124033Z.json`.

### 8.9 V7 residual + LayerNorm paired ablation

RTX 5090, PyTorch `2.13.0+cu130`, GPU vật lý #1 idle, FP32 public/FP16
internal, `torch.compile(mode="max-autotune")`, accuracy trials `5`, warmup
`50`, repeats `200`, rounds `5`:

| Shape/order | V4.3 median | V7a median | V4.3 GPU time | V7a GPU time | Evidence |
|---|---:|---:|---:|---:|---|
| #2, V4.3 → V7a | 0.0689 ms | 0.0699 ms | 0.0546 ms | 0.0546 ms | 32 kernels, 28 Triton cả hai |
| #8, V4.3 → V7a | 2.5155 ms | 2.5235 ms | 2.4659 ms | 2.4713 ms | 32 kernels, 28 Triton cả hai |
| #12, V4.3 → V7a | 0.0813 ms | 0.0792 ms | 0.0774 ms | 0.0771 ms | 33 kernels, 29 Triton cả hai |
| #12, V7a → V4.3 | 0.0813 ms | 0.0812 ms | 0.0775 ms | 0.0778 ms | Effect đảo dấu theo device time |

Kernel names giống nhau; V7a chỉ diễn đạt lại source graph mà không thay
compiled execution. Do đó không có performance evidence để thay V4.3.

### 8.10 V8 fused FFN/GELU paired ablation

RTX 5090, PyTorch `2.13.0+cu130`, GPU vật lý #1 idle, FP32 public/FP16
internal, `torch.compile(mode="max-autotune")`. Paired #6 chốt dùng accuracy
trials `5`, warmup `10`, repeats `20`, rounds `5`; các lượt order check trước
dùng accuracy `3`, repeats `20`, rounds `3`.

| Shape/order | V4.3 median | V8 median | V4.3 GPU time | V8 GPU time | Evidence |
|---|---:|---:|---:|---:|---|
| #6, V4.3 → V8 | 26.6799 ms | 25.4092 ms | 26.6680 ms | 25.3948 ms | 32 → 29 kernels; 28 → 21 Triton events |
| #6, V4.3 → V8, order check | 26.6517 ms | 25.3846 ms | — | — | `-4.75%` latency |
| #6, V8 → V4.3 | 26.6879 ms | 25.3693 ms | — | — | `-4.94%` latency |
| #8, three-order check | 2.5145–2.5341 ms | 2.5145–2.5183 ms | 2.4649–2.4732 ms | 2.4652–2.4710 ms | Effect quanh zero; fallback V4.3 |

Full V8 #1–#13 matrix (`accuracy=5`, warmup/repeats/rounds `20/100/3`) PASS
tất cả shape, geomean speedup `9.775x`; #6 đạt `25.3436 ms`. Paired #6 là
performance evidence chính vì so candidate trong cùng process; matrix xác nhận
dispatcher không làm hỏng correctness hoặc các shape fallback. Artifacts:
`profile-results/profile_shape06_20260828T151245Z.json` và
`benchmark-results/matrix_v8_FusedFFNGELU_float32_20260828T151419Z.json`.

### 8.11 V8.1 force-all paired comparison

RTX 5090, PyTorch `2.13.0+cu130`, GPU vật lý #1 idle, FP32 public/FP16
internal, max-autotune. #1–#5/#7–#13 dùng accuracy/warmup/repeats/rounds
`5/20/100/3` theo cả hai implementation orders. #6 exact-wrapper run dùng
`5/10/20/3`; reverse-order evidence kế thừa V8 vì code path giống hệt.

| Shape | V4.3→V8.1 latency Δ | V8.1→V4.3 latency Δ | Kết luận |
|---:|---:|---:|---|
| 1 | -2.00% | -1.57% | Win nhỏ, giữ dấu |
| 2 | +0.00% | +2.93% | Host regression/order-sensitive |
| 3 | -1.42% | -1.47% | Win nhỏ, giữ dấu |
| 4 | -2.73% | -2.69% | Win giữ dấu |
| 5 | -1.52% | -1.21% | Win nhỏ, giữ dấu |
| 6 | -4.78% | -4.94% | Win rõ và ổn định |
| 7 | +0.05% | -1.49% | Host quantization/noise |
| 8 | -0.40% | -0.56% | Dưới 1% |
| 9 | -0.06% | -1.42% | Order-sensitive host effect |
| 10 | -1.49% | -1.51% | Win nhỏ, giữ dấu |
| 11 | -0.09% | -0.16% | Raw GPU regress 0.41–0.52% |
| 12 | -2.52% | +2.47% | Đổi dấu; không phải win |
| 13 | -0.66% | -0.92% | Dưới 1% |

Geomean latency giảm `1.365%` và `0.969%` theo hai orders. V8.1 thường giảm
3–4 GPU kernels nhưng không Pareto-win mọi shape, nên không thay V8 dispatcher.
Representative artifacts:
`profile-results/profile_shape01_20260828T162815Z.json`,
`profile-results/profile_shape01_20260828T163337Z.json`,
`profile-results/profile_shape12_20260828T163158Z.json`,
`profile-results/profile_shape12_20260828T163717Z.json`, và
`profile-results/profile_shape06_20260828T163922Z.json`.

### 8.12 V9 persistent full-MLP ablation

RTX 5090, PyTorch `2.13.0+cu130`, GPU vật lý #1 idle, FP32 public/FP16
internal, max-autotune. Isolated FFN CUDA-event sweep cho speedup từ `1.179x`
(#13 mapping) tới `1.589x` (#6 mapping), failed=0. Whole-model paired results:

| Shape | V4.3 | V9 | V9 latency Δ | Raw GPU evidence |
|---:|---:|---:|---:|---|
| 2 | 0.0689 ms | 0.0689 ms | 0.0% | V9 +1.1% |
| 12 | 0.0812 ms | 0.0812 ms | 0.0% | V9 -0.6% |
| 7, first order | 0.0709 ms | 0.0699 ms | -1.4% | V9 -12.3% |
| 7, reverse | 0.0692 ms | 0.0709 ms | +2.5% | V9 vẫn -12.3%; primary đổi dấu |
| 1 | 0.1325 ms | 0.1387 ms | +4.7% | V9 +5.2% |
| 5 | 0.2084 ms | 0.2126 ms | +2.0% | V9 +1.5% |
| 13 | 1.0891 ms | 1.1463 ms | +5.3% | V9 +4.9% |

Shape #6 triple paired cho V4.3/V8/V9 lần lượt
`26.6497/25.4071/25.4481 ms`, raw GPU `26.6522/25.3974/25.4512 ms`, kernels
`32/29/25`. V9 không cải thiện current best path dù fuse nhiều hơn. Artifacts:
`profile-results/profile_shape07_20260828T171520Z.json`,
`profile-results/profile_shape07_20260828T171738Z.json`, và
`profile-results/profile_shape06_20260828T171913Z.json`.

### 8.13 Historical non-official diagnostics

| Config | Candidate | Baseline median | Candidate median | Speedup |
|---|---|---:|---:|---:|
| Default non-causal | V1 | 1.3569 ms | 1.3112 ms | 1.035x |
| Default non-causal | V2 | 1.3767 ms | 0.9951 ms | 1.384x |
| Default non-causal | V3 | 1.3758 ms | 0.7724 ms | 1.781x |

Các số này chỉ giải thích ablation trong quá trình phát triển và không được dùng làm benchmark submission.

### 8.14 V14 optimized-only diagnostic trên official shape #14

Accuracy gate đã PASS trước timing. Original baseline latency và speedup đều
N/A vì explicit score cần khoảng `18.6 TiB`; không dùng query-blocked accuracy
reference để thay baseline performance. Một warmup và năm CUDA Event repeats:

| Baseline | V14 median / p90 / min | Mean | Throughput | Timed peak | Speedup |
|---|---:|---:|---:|---:|---:|
| N/A (OOM by construction) | 6683.9873 / 6700.7756 / 6656.8384 ms | 6682.2406 ms | 478,756.15 token/s | 27.000 GiB | N/A |

Output được `del`, Python cycles được collect và CUDA cache được cleanup giữa
repeat, sau khi Event đã synchronize; lifecycle work nằm ngoài vùng timing.
Đây là exact-shape optimized-only diagnostic, không phải paired speedup.

### 8.15 V15 direct-layout QKV trên official shape #13

RTX 5090 physical index `1` idle, PyTorch `2.13.0+cu130`, FP32 public/FP16
internal, `torch.compile(mode="max-autotune")`. Strict gate năm trial PASS
`0/41,943,040`, max abs `0.00147235`. Paired measurement dùng warmup `50`,
repeats `200`, rounds `5`, profile warmup/iterations `50/20`:

| Order | V14.1 median / p90 | V15 median / p90 | Latency Δ | V14.1 raw GPU | V15 raw GPU | Raw Δ |
|---|---:|---:|---:|---:|---:|---:|
| V14.1 → V15 | 1.1080 / 1.1166 ms | **1.0971 / 1.1049 ms** | **-0.98%** | 1.0966 ms | **1.0838 ms** | **-1.17%** |
| V15 → V14.1 | 1.1251 / 1.1330 ms | **1.1003 / 1.1085 ms** | **-2.20%** | 1.1062 ms | **1.0867 ms** | **-1.76%** |

Baseline medians nằm trong `41.7740–41.7829 ms`; V15 đạt `38.077x` và
`37.965x` theo process tương ứng. Cả hai orders giữ dấu, nên D-031 promote
candidate. Cả hai compiled graphs có 29 GPU kernels, hai memory events, một
compiled region và một CUDA Graph launch; Triton GPU events giảm `21 → 17`.
Flash device time gần như không đổi, do đó đây là QKV/projection-layout gain,
không phải Flash-kernel speedup.

Artifacts chính:
`profile-results/profile_shape13_20260830T034815Z.json` và
`profile-results/profile_shape13_20260830T035041Z.json`. Official #2/#12
fallback canaries nằm trong
`benchmark-results/matrix_v15_DirectQKVLayout_float32_20260830T035331Z.json`.

### 8.16 V16 compiled executor trên official shape #14

RTX 5090 idle, PyTorch `2.11.0+cu128`, FP32 public/FP16 internal,
`torch.compile(mode="max-autotune")` chỉ cho thân B=1. Full strict gate đủ 32
batch PASS `0/3,276,800,000`, max abs `0.000944197`, elapsed `332.308 s`, peak
accuracy `19.585 GiB`.

| Order | V14.1 median | V16 median | V16 latency Δ | V14.1 peak | V16 peak |
|---|---:|---:|---:|---:|---:|
| V14.1 → V16 | 7396.7202 ms | **7166.8359 ms** | **-3.11%** | 26.977 GiB | **24.487 GiB** |
| V16 → V14.1 | 7435.5688 ms | **7166.8359 ms** | **-3.61%** | 26.977 GiB | **24.487 GiB** |

Đây là sandwich measurement bằng process riêng với cùng seed/warmup/repeats;
V16 sample nằm giữa hai controls. Original baseline không executable nên
speedup chính thức vẫn N/A. #2/#13 fallback canaries PASS strict; timing
`1/1/1` của chúng không dùng làm performance claim.
Artifact: `benchmark-results/shape14_v16_v141_sandwich_20260830.json`.

### 8.17 V17 batch-chunk B=2 trên official shape #14

Cùng Vast.ai RTX 5090/PyTorch `2.11.0+cu128`. Full strict gate chạy executor
B=2 PASS `0/3,276,800,000`, max abs `0.000944197`, mean abs
`6.56366e-05`, elapsed `333.542 s`, accuracy peak `20.348 GiB`.

| Sequence | Median / p90 | V17 Δ so adjacent V16 | Timed peak |
|---|---:|---:|---:|
| V16 trước | 7183.8022 / 7202.1914 ms | — | 24.487 GiB |
| V17 lần 1 | **7141.3345 / 7151.5054 ms** | `-0.59%` / `-0.30%` | 24.487 GiB |
| V16 sau | 7162.9731 / 7174.2969 ms | — | 24.487 GiB |
| V17 lần 2 | **7131.5425 / 7148.6027 ms** | `-0.44%` | 24.487 GiB |

Trung bình hai median V16/V17 là `7173.3877/7136.4385 ms`, chênh
`-0.515%`. Gain giữ dấu nhưng dưới 1% và chưa vượt confidence threshold so
với drift trong từng process. D-033 không promote V17; original baseline và
speedup vẫn N/A. Artifact:
`benchmark-results/shape14_v17_v16_alternating_20260830.json`.

### 8.18 Shape-#14 profiler và attention backend shootout

Cùng Vast.ai RTX 5090, PyTorch `2.11.0+cu128`, CUDA 12.8, TF32 và
max-autotune. Đây là **inner-executor/isolated diagnostic**, không phải paired
official speedup vì original baseline #14 cần khoảng `18.6 TiB` attention
scores và không chạy trên GPU 32 GiB.

| Diagnostic | Control | Candidate/result | Kết luận |
|---|---:|---:|---|
| V16 inner B=1 | — | `218.4892 ms/call` | attention `199.0931 ms` = `92.258%` raw device |
| V17 inner B=2 | V16 `218.4892 ms/sample` | `218.0144 ms/sample` | batching chỉ `~0.22%` inner gain |
| Built-in SDPA sandwich | Flash `217.3188/218.4642 ms` | cuDNN `223.6575 ms` | cuDNN chậm `2.38–2.92%` |
| Efficient SDPA | Flash khoảng `218 ms` | `417.5209 ms` | gần `2x` chậm |
| FA4 `4.0.0b28` attention | PyTorch Flash `100.9365 ms` | `108.7358 ms` | strict PASS nhưng chậm `7.72%` |
| Sage PV-FP16 attention | PyTorch Flash `100.5045 ms` | `72.1337 ms` | `1.3933x` nhưng strict FAIL `94/102.4M` |

FA4 bị reject theo performance. Sage được phép đi tiếp sang version V18 vì
upside lớn, nhưng eager full-model B=1 vẫn fail `1/102.4M`; theo accuracy-first
gate không có V18 model timing. Artifacts có prefix
`profile-results/shape14_inner_*`, `shape14_fa4_probe_*` và
`shape14_sage_probe_*`; D-034 ghi quyết định giữ V16.

### 8.19 V15.1 direct-layout QKV cross-shape sweep

Cùng Vast.ai RTX 5090/PyTorch `2.11.0+cu128`, official #1–#12 PASS strict năm
trial trước timing. Paired max-autotune dùng warmup/repeats/rounds `20/100/3`
và chạy cả V15→V15.1 lẫn V15.1→V15. Delta là `V15.1/V15 - 1`:

| Shape | End-to-end order A / B | Raw GPU order A / B | Kết quả |
|---:|---:|---:|---|
| #1 | `-1.27% / +0.64%` | `+2.83% / +6.61%` | Reject |
| #2 | `-0.60% / -0.65%` | `+26.82% / +6.94%` | Retest dài: reject |
| #3 | `+0.67% / 0.00%` | `+15.88% / +15.59%` | Reject |
| #4 | `-1.88% / -1.27%` | `-1.05% / +2.60%` | Retest dài: reject |
| #5 | `+1.84% / +2.68%` | `+1.82% / +2.71%` | Reject |
| #6 | **`-2.41% / -4.44%`** | **`-3.03% / -2.96%`** | **Winner** |
| #7 | `0.00% / +0.68%` | `+14.89% / +16.02%` | Reject |
| #8 | `+2.76% / +3.31%` | `+2.18% / +3.22%` | Reject |
| #9 | `-0.03% / -0.04%` | `+1.58% / -0.09%` | Neutral/reject |
| #10 | `-0.02% / +0.62%` | `+5.11% / +8.76%` | Reject |
| #11 | `+4.34% / +3.32%` | `+3.98% / +2.99%` | Reject |
| #12 | `-1.27% / +2.60%` | `+3.75% / -0.47%` | Reject |

#6 giảm geometric two-order ratio `3.43%` end-to-end và `2.99%` raw GPU.
Shape #2/#4 được retest cả hai orders bằng `100/1000/7`; raw GPU vẫn regress
lần lượt `16.10–17.62%` và `5.52–6.17%`, loại các apparent host wins. D-035
giữ V15.1 là ablation và V16 là main ở thời điểm đó; D-038 sau này promote
V16.1 nhưng vẫn chưa thêm #6 dispatch.
Artifacts chính là `matrix_v15_1_DirectQKVAll_float32_20260830T142400Z.json`
và `profile_shape{01..12}_20260830T143*/T144*.json`.

### 8.20 Source-clean V16.1 main

Official #13 strict năm trial PASS `0/41,943,040`, max abs `0.00147235`.
Paired V14.1↔V16.1 dùng warmup/repeats/rounds `20/100/3`:

| Order | V14.1 median | V16.1 median | Delta | Graph |
|---|---:|---:|---:|---|
| V14.1 → V16.1 | 1.1977 ms | 1.2089 ms | +0.93% | cùng 29 kernels / 21 Triton events |
| V16.1 → V14.1 | 1.2219 ms | 1.2099 ms | -0.98% | cùng 29 kernels / 21 Triton events |

Đây là equivalence/control evidence, không phải speedup claim. V16.1 không qua
V15 và chứa standalone executor #14. Exact-config compiled B=1 canary PASS
`0/102.4M`, max abs `0.000719786`. Artifacts #13:
`matrix_v16_1_NoDirectQKV13_float32_20260830T151525Z.json` và
`profile_shape13_20260830T151536Z/151623Z.json`.

### 8.21 V17-Sage corrected #1–#13 rejection

RTX 5090/PyTorch `2.11.0+cu128`, SageAttention 2.2.0, TF32, seed `1234`, năm
accuracy trial và max-autotune `20/100/3`:

| ID | Accuracy | Baseline ms | Optimized ms | Speedup |
|---:|---|---:|---:|---:|
| 1 | PASS | 2.6558 | 2.1926 | 1.211x |
| 2 | PASS | 2.5521 | 2.1171 | 1.205x |
| 3 | PASS | 2.6572 | 2.1284 | 1.248x |
| 4 | PASS | 2.5783 | 2.1284 | 1.211x |
| 5 | PASS | 2.6107 | 2.1669 | 1.205x |
| 6 | **FAIL** | — | — | — |
| 7 | PASS | 2.5814 | 2.1648 | 1.192x |
| 8 | PASS, fallback | 6.7651 | 2.9503 | 2.293x |
| 9 | **FAIL** | — | — | — |
| 10 | PASS | 2.5973 | 1.8344 | 1.416x |
| 11 | PASS | 2.6936 | 2.1472 | 1.254x |
| 12 | PASS, fallback | 2.6152 | 0.1618 | 16.161x |
| 13 | PASS | 41.9384 | 2.4573 | 17.067x |

#6/#9 max abs lần lượt `0.00250164`/`0.00255397` và strict comparator báo
FAIL, nên aggregate performance không hợp lệ. #13 cho thấy Sage có raw upside
ở long sequence, còn S=128 chỉ khoảng `1.19–1.42x`; V18-Sage direct được tạo
để tách correction overhead nhưng vẫn chỉ là performance diagnostic.

D-038 dùng evidence này để promote V16.1 vì source cleanliness. Promotion không
đổi các số lịch sử thành performance evidence mới và chấp nhận bỏ V15/V16 win
`0.98–2.20%` ở #13. Fresh standalone evidence được ghi riêng bên dưới.

### 8.22 Final standalone V16.1 official results

Ngày 2026-08-31, commit
`4f77a04fd51c3a2ad8b9d8986657915ae3ca94d6` được chạy lại trên đúng
`main.py → v16_1_clean.py`. #1–#13 dùng năm accuracy trial, warmup/repeats/rounds
`20/100/3`, baseline eager và optimized `max-autotune`; mọi row strict PASS,
failed `0`, worst max abs `0.00179085`, geomean speedup **7.904x**.

| ID | Baseline median / p90 ms | Optimized median / p90 ms | Speedup | Max abs |
|---:|---:|---:|---:|---:|
| 1 | 0.8146 / 0.8167 | 0.1389 / 0.1422 | 5.863x | 0.00127272 |
| 2 | 0.7136 / 1.0536 | 0.0566 / 0.0575 | 12.606x | 0.00114284 |
| 3 | 0.7764 / 0.7855 | 0.0607 / 0.0611 | 12.789x | 0.00123456 |
| 4 | 0.7214 / 0.7289 | 0.0771 / 0.0775 | 9.354x | 0.00134718 |
| 5 | 1.3092 / 1.3322 | 0.2208 / 0.2290 | 5.930x | 0.00147235 |
| 6 | 179.0890 / 179.3125 | 26.5491 / 26.8043 | 6.746x | 0.00160612 |
| 7 | 0.7208 / 0.7353 | 0.0732 / 0.0734 | 9.853x | 0.00179085 |
| 8 | 7.1229 / 7.1628 | 2.8616 / 2.8791 | 2.489x | 0.00134873 |
| 9 | 0.6540 / 0.6646 | 0.1574 / 0.1606 | 4.155x | 0.00126606 |
| 10 | 0.7254 / 0.7347 | 0.1451 / 0.1467 | 5.001x | 0.00134726 |
| 11 | 1.8169 / 1.8390 | 0.1996 / 0.2036 | 9.105x | 0.00140083 |
| 12 | 0.7213 / 0.7279 | 0.0854 / 0.0857 | 8.442x | 0.00134718 |
| 13 | 42.2520 / 42.2904 | 1.2455 / 1.2544 | 33.925x | 0.00147235 |

Full shape #14 chạy đúng `B=32` qua memory-bounded reference: strict PASS
`0/3,276,800,000`, max abs `0.000944197`, mean abs `6.56367e-05`, elapsed
`348.534 s`, peak accuracy allocation `19.967 GiB`. Optimized-only CUDA Event
samples là `7135.0098/7171.5771/7213.5254/7239.6484/7260.9688 ms`; median
`7213.5254 ms`, p90 `7252.4406 ms`, throughput `443,611.11 token/s`, peak
`24.487 GiB`. Baseline latency/speedup là N/A vì explicit score cần khoảng
`18.6 TiB`.

Raw JSON/CSV, full #14 logs và environment manifest được check in tại
`results/cross-host-driver580/`. Đây là historical cross-host evidence; section
tiếp theo chứa promoted driver-595 timeline.

### 8.23 Full checkpoint timeline và promoted driver-595 final

Cùng source snapshot `4f77a04`, toàn bộ checkpoint chính được chạy lại trên
host RTX 5090 driver `595.71.05` theo đúng strict accuracy `5` trial và timing
`20/100/3`. V16.1 được đặt làm start/end control trước khi sweep; headline lấy
start-control nếu drift gate 3% PASS.

| Checkpoint | Accuracy | Forward geomean | Reverse geomean |
|---|---:|---:|---:|
| Baseline | 13/13 PASS | 1.0011x | — |
| V1 | 13/13 PASS | 1.0763x | — |
| V2 | 13/13 PASS | 1.4353x | — |
| V3.1 eager | 13/13 PASS | 2.1006x | — |
| V3.1 compiled | 0/13, timing skipped | N/A | — |
| V4.1 | 13/13 PASS | 10.1999x | 10.1926x |
| V4.2 | 13/13 PASS | 10.4489x | 10.4349x |
| V4.3 | 13/13 PASS | 11.6755x | 11.6948x |
| V8 | 13/13 PASS | 11.7854x | 11.6580x |
| V11 | 13/13 PASS | 11.7483x | 11.7439x |
| V16.1 start | 13/13 PASS | **11.8030x** | 11.7617x |
| V16.1 end | 13/13 PASS | 11.8383x | — |

V16.1 baseline/optimized geomean drift lần lượt `0.166%`/`0.458%`; heavy
#6/#8/#13 đều PASS budget, max `1.042%`. V3.1 compiled fail strict cả 13 shapes,
tổng `201,682` failed elements; correctness gate đã skip toàn bộ timing.

Promoted V16.1 start-control:

| ID | Baseline median ms | Optimized median ms | Speedup | Max abs |
|---:|---:|---:|---:|---:|
| 1 | 1.7640 | 0.1343 | 13.134x | 0.00127273 |
| 2 | 1.7764 | 0.1108 | 16.035x | 0.00114284 |
| 3 | 1.7882 | 0.1108 | 16.142x | 0.00124183 |
| 4 | 1.7384 | 0.1118 | 15.549x | 0.00134721 |
| 5 | 1.7535 | 0.2102 | 8.340x | 0.00147235 |
| 6 | 177.3218 | 25.1813 | 7.042x | 0.00160612 |
| 7 | 1.7424 | 0.1118 | 15.584x | 0.00179085 |
| 8 | 6.6464 | 2.7936 | 2.379x | 0.00134873 |
| 9 | 1.5747 | 0.1513 | 10.409x | 0.00126606 |
| 10 | 1.7550 | 0.1404 | 12.498x | 0.00134724 |
| 11 | 1.7345 | 0.1860 | 9.326x | 0.00140083 |
| 12 | 1.7501 | 0.1098 | 15.940x | 0.00134721 |
| 13 | 41.8362 | 1.0793 | 38.762x | 0.00147235 |

Shape #14 theo scope cuối chỉ gồm Baseline/V16.1. Baseline là
`INFEASIBLE_STATIC`, latency/speedup N/A. V16.1 B1 PASS `0/102,400,000`;
streamed B32 PASS `0/3,276,800,000`, max abs `0.000944197`; native B32 output
`[32,100000,1024]` FP32 PASS. Optimized-only samples
`6987.4644/6983.9238/6987.4033/6992.8545/6994.9302 ms`; median
`6987.4644 ms`, p90 `6994.0999 ms`, throughput `457,962.98 token/s`, peak
`24.487 GiB`.

Driver-595 baseline geomean `3.5108 ms` chậm hơn driver-580 `2.0355 ms`
`72.48%`, còn optimized `0.29745 ms` chậm hơn `0.25752 ms` `15.51%`. Vì code
revision không đổi, `7.904x → 11.803x` là host-ratio effect, không phải code
improvement. Curated evidence nằm trong `results/timeline-rtx5090-driver595/`
và `results/final/`.

## 9. Cách tái lập

### 9.1 Chuẩn bị môi trường

Final evidence dùng Python `3.12.14`, PyTorch `2.11.0+cu128`, CUDA wheel
`12.8`, cuDNN `9.19.0` và Triton `3.6.0`. Trên RTX 50/`sm120`, dùng PyTorch
wheel CUDA 12.8 trở lên.

```bash
cd /workspace/techjam-2026-track3
/venv/main/bin/python -c \
  "import torch,triton; print(torch.__version__, torch.version.cuda, triton.__version__, torch.cuda.is_available())"
```

Với môi trường mới, tạo venv riêng và cài PyTorch build tương thích GPU; ghi
exact versions vào artifact kết quả. Không trộn kết quả từ hai software stack
vào cùng final aggregate.

### 9.2 Kiểm tra syntax

```bash
python -m py_compile main.py matrix_runner.py profile_models.py \
  torch_transformer_benchmark.py v16_1_clean.py shape14_accuracy.py \
  shape14_optimized_benchmark.py shape14_profile.py \
  shape14_fa4_probe.py shape14_sage_probe.py timeline_adapter.py \
  timeline_runner.py shape14_checkpoint_worker.py shape14_timeline_runner.py
python timeline_runner.py --list-checkpoints
python timeline_runner.py --list-shapes
python shape14_timeline_runner.py --list-checkpoints
```

### 9.3 Chạy timeline đủ #1–#13

```bash
CUDA_VISIBLE_DEVICES=1 python timeline_runner.py \
  --checkpoints v16_1,baseline,v1,v2,v3_1_eager,v3_1_compiled,v4_1,v4_2,v4_3,v8,v11,v16_1 \
  --shape-ids 1-13 --device cuda:0 --dtype float32 \
  --accuracy-trials 5 --warmup 20 --repeats 100 \
  --benchmark-rounds 3 --seed 1234 --timeout 1800 \
  --compile-mode max-autotune --control-drift-threshold 0.03
```

Runner chạy từng checkpoint/shape trong process riêng, tiếp tục qua
OOM/error/timeout và ghi partial JSON/CSV sau mỗi case. V16.1 đầu/cuối là
control cho drift gate 3%; shape #14 cố ý không thuộc runner này.

Sau D-042, active aliases chỉ còn `main`, `best`, `v16.1` và các spelling
`v16_1`/`v16.1.clean`/`v16_1_clean`; tất cả resolve qua `main.py` tới
`v16_1_clean.py`. Timeline adapter resolve checkpoint cũ trực tiếp từ
`archive/versions/`. Dùng `--shape-ids 1,2,7` để rerun một diagnostic subset;
default của timeline runner là đủ #1–#13.

Bảng terminal tổng kết `status`, `max_abs`, baseline median, optimized median và speedup cho từng shape. `max_abs` giúp nhìn nhanh error margin nhưng không thay thế correctness gate: PASS vẫn yêu cầu mọi phần tử thỏa strict `absolute < 0.002 OR relative < 0.02`.

Runner đã PASS local smoke trên official shape #2 và GPU smoke trên official shape #12. Các smoke run dùng rất ít warmup/repeats nên chỉ xác nhận orchestration/parser, không được đưa vào bảng performance.

### 9.3.1 Baseline/V16.1 riêng cho shape #14

```bash
CUDA_VISIBLE_DEVICES=1 python shape14_timeline_runner.py \
  --checkpoints baseline,v16_1 --device cuda:0 --seed 1234 \
  --batch-limit 32 --query-chunk 256 --compare-token-chunk 2048 \
  --warmup 1 --repeats 5 --compile-mode max-autotune
```

Baseline được ghi `INFEASIBLE_STATIC`; chỉ V16.1 chạy B1, full streamed B32,
native B32 rồi optimized-only timing. Không dùng standard matrix runner cho
performance #14: original baseline không executable và không có paired speedup
hợp lệ.

V16.1 tự compile riêng sample executor; không cần `--compile-user` cho loop ngoài.
Các predecessor control đã chuyển vào archive và không còn là active alias.

Các mục §9.4–§9.26 bên dưới lưu command lịch sử trước D-042. Khi cần tái lập,
dùng source trong `archive/versions/` và thêm cả root lẫn thư mục archive vào
`PYTHONPATH`; các command đó không mô tả active runner hiện tại.

### 9.4 Profile v1/v2/v3 trên cùng official shape

```bash
CUDA_VISIBLE_DEVICES=1 python profile_models.py \
  --impl v1 v2 v3 --shape-id 1
```

Command chạy mỗi implementation trong subprocess riêng, giữ correctness gate trước performance và đo end-to-end latency bằng CUDA Event. Eager path thu ATen operator cùng model-stage breakdown bằng PyTorch Profiler/Kineto; model-stage table tách norm1, QKV projection, view/reshape, attention core, output projection, residual, norm2, FFN in/GELU/out, masking/copy và final norm. Terminal in bảng so sánh; JSON nằm trong `profile-results/`. Muốn xuất Chrome trace để mở bằng Perfetto, thêm `--export-traces --record-shapes`.

Để profile compiled path và xác nhận CUDA Graph/Triton evidence:

```bash
TORCH_LOGS=perf_hints CUDA_VISIBLE_DEVICES=1 python profile_models.py \
  --impl v3.1 --shape-id 1 \
  --compile-user --compile-mode reduce-overhead \
  --export-traces --record-shapes
```

Compiled terminal/JSON báo raw GPU device time, kernel/memory-event count, Triton events/launches, `Torch-Compiled Region`, CUDA Graph launches, top GPU event names và steady/peak CUDA allocation. Eager ATen stage scopes bị tắt có chủ đích trên compiled path để tránh thay đổi hoặc recompile graph. Muốn tách gain của Inductor whole-graph optimization khỏi cấu hình giảm overhead, chạy lại cùng command với `--compile-mode default` rồi so với `reduce-overhead`.

`attention_core` vẫn là một fused SDPA stage. Phép `QKᵀ`, scale, causal/key mask, softmax và `probabilities @ V` xảy ra bên trong cùng backend kernel nên không được báo như các timing độc lập; de-fuse chúng chỉ tạo explicit diagnostic khác với implementation đang benchmark.

Profiler dùng cùng fixed input và `valid_token_mask` với benchmark harness. Khi `padding_ratio=0`, mask vẫn là tensor all-valid thay vì `None`, nhờ đó profile phản ánh đúng đường thực thi được đo bởi benchmark.

### 9.5 Benchmark riêng official shape #1 với v1

```bash
CUDA_VISIBLE_DEVICES=1 python v1_fuseQKV.py \
  --device cuda:0 --dtype float32 \
  --batch-size 64 --seq-len 128 --d-model 128 \
  --heads 4 --ffn-dim 128 --layers 4 --causal
```

### 9.6 Benchmark riêng official shape #1 với v2

```bash
CUDA_VISIBLE_DEVICES=1 python v2_SPDA.py \
  --device cuda:0 --dtype float32 \
  --batch-size 64 --seq-len 128 --d-model 128 \
  --heads 4 --ffn-dim 128 --layers 4 --causal
```

### 9.7 Benchmark riêng official shape #1 với v3

```bash
CUDA_VISIBLE_DEVICES=1 python v3_SDPA_NoCopy.py \
  --device cuda:0 --dtype float32 \
  --batch-size 64 --seq-len 128 --d-model 128 \
  --heads 4 --ffn-dim 128 --layers 4 --causal \
  --accuracy-trials 3 \
  --warmup 20 --repeats 100 --benchmark-rounds 3
```

Padding, non-causal và shape tự tạo chỉ được chạy như correctness/debug cases, không ghi nhận performance. Có thể đổi `--dtype` để kiểm tra fallback nếu benchmark chính thức yêu cầu dtype đó.

### 9.8 Benchmark official shapes #1 và #13 với v3.1

```bash
CUDA_VISIBLE_DEVICES=1 python matrix_runner.py \
  --impl v3.1 --shape-ids 1,13 --device cuda:0 --dtype float32
```

### 9.9 Profile V4.1 eager và compiled

```bash
CUDA_VISIBLE_DEVICES=1 python profile_models.py \
  --impl v3_1_CausalMask.py v4_FP16.py v4_1_FP16_GELU.py \
  --shape-id 1 --accuracy-trials 5 \
  --warmup 50 --repeats 200 --benchmark-rounds 5

CUDA_VISIBLE_DEVICES=1 python profile_models.py \
  --impl v3_1_CausalMask.py v4_FP16.py v4_1_FP16_GELU.py \
  --shape-id 1 --accuracy-trials 5 \
  --warmup 50 --repeats 200 --benchmark-rounds 5 \
  --compile-user --compile-mode reduce-overhead
```

Runner alias của candidate là `v4.1.fp16`, ví dụ:

```bash
CUDA_VISIBLE_DEVICES=1 python matrix_runner.py --impl v4.1.fp16
```

### 9.10 Chạy V4.2 dispatcher trên shapes #1–#13

```bash
CUDA_VISIBLE_DEVICES=1 python matrix_runner.py \
  --impl v4.2.dispatch \
  --shape-ids 1,2,3,4,5,6,7,8,9,10,11,12,13 \
  --compile-user --compile-mode reduce-overhead
```

Shape #14 được để pending có chủ đích trong lượt phát triển này, không bị xóa
khỏi `SHAPES` hoặc correctness requirement.

### 9.11 Chạy V4.3 Flash-first trên shapes #1–#13

```bash
CUDA_VISIBLE_DEVICES=1 python matrix_runner.py \
  --impl v4.3 \
  --shape-ids 1,2,3,4,5,6,7,8,9,10,11,12,13 \
  --compile-user --compile-mode max-autotune
```

Alias `v4.3` và `v4.3.flash` đều trỏ tới `v4_3_Flash.py`.
Forced/static V4.3 ablation files đã được xóa sau cleanup. Shape #14 vẫn nằm
trong runner và phải được xử lý trước final.

### 9.12 Tái lập V5.1 FP16 accumulation ablation

```bash
CUDA_VISIBLE_DEVICES=1 python matrix_runner.py \
  --impl v5.1 --shape-ids 8,10 \
  --accuracy-trials 5 \
  --compile-user --compile-mode max-autotune
```

Command mặc định dừng từng row trước benchmark nếu accuracy fail. Không thêm
`--benchmark-on-failure` để tạo performance claim cho candidate sai.

### 9.13 Tái lập V6 approximate GELU

```bash
CUDA_VISIBLE_DEVICES=1 python matrix_runner.py \
  --impl v6 --shape-ids 1,2,3,4,5,6,7,8,9,10,11,12,13 \
  --accuracy-trials 5 \
  --compile-user --compile-mode max-autotune
```

Chỉ dùng latency khi GPU vật lý #1 idle. Matrix ngày 2026-08-28 xác nhận
accuracy nhưng performance bị invalid bởi concurrent training workload.

### 9.14 Tái lập V7 residual + LayerNorm pipeline

```bash
TORCH_LOGS=perf_hints CUDA_VISIBLE_DEVICES=1 python profile_models.py \
  --impl v4_3_Flash.py v7_ResidualLayerNorm.py \
  --shape-id 12 --accuracy-trials 5 \
  --warmup 50 --repeats 200 --benchmark-rounds 5 \
  --compile-user --compile-mode max-autotune
```

Đảo thứ tự hai `--impl` để kiểm tra order bias. Candidate không có custom
Triton; mục tiêu của command là tái lập compiled-graph equivalence.

### 9.15 Tái lập V8 fused FFN/GELU

Paired performance trên official #6:

```bash
TORCH_LOGS=perf_hints CUDA_VISIBLE_DEVICES=1 python profile_models.py \
  --impl v4.3 v8 --shape-id 6 --accuracy-trials 5 \
  --warmup 10 --repeats 20 --benchmark-rounds 5 \
  --compile-user --compile-mode max-autotune
```

Accuracy/performance matrix #1–#13:

```bash
TORCH_LOGS=perf_hints CUDA_VISIBLE_DEVICES=1 python matrix_runner.py \
  --impl v8 --shape-ids 1,2,3,4,5,6,7,8,9,10,11,12,13 \
  --accuracy-trials 5 --warmup 20 --repeats 100 --benchmark-rounds 3 \
  --compile-user --compile-mode max-autotune
```

### 9.16 Tái lập V8.1 force-all comparison

Chạy hai orders để kiểm tra order bias:

```bash
for sid in 1 2 3 4 5 7 8 9 10 11 12 13; do
  TORCH_LOGS=perf_hints CUDA_VISIBLE_DEVICES=1 python profile_models.py \
    --impl v4.3 v8.1 --shape-id "$sid" --accuracy-trials 5 \
    --warmup 20 --repeats 100 --benchmark-rounds 3 \
    --compile-user --compile-mode max-autotune
done
```

Đảo `--impl v8.1 v4.3` cho direction thứ hai. #6 dùng
`--warmup 10 --repeats 20 --benchmark-rounds 3` để giới hạn thời gian của
baseline batch 10000.

### 9.17 Tái lập V9 persistent MLP

Accuracy matrix:

```bash
TORCH_LOGS=perf_hints CUDA_VISIBLE_DEVICES=1 python matrix_runner.py \
  --impl v9 --shape-ids 1,2,3,4,5,6,7,8,9,10,11,12,13 \
  --accuracy-trials 5 --warmup 1 --repeats 1 --benchmark-rounds 1 \
  --compile-user --compile-mode max-autotune
```

Paired representative shape; đảo thứ tự implementations để kiểm tra bias:

```bash
TORCH_LOGS=perf_hints CUDA_VISIBLE_DEVICES=1 python profile_models.py \
  --impl v4.3 v9 --shape-id 7 --accuracy-trials 5 \
  --warmup 50 --repeats 200 --benchmark-rounds 5 \
  --compile-user --compile-mode max-autotune
```

### 9.18 Tái lập V11/V14.1/V15/V16 rollback và V16.1 main

`main.py`, alias `main` và alias `best` chạy promoted V16.1; alias `v16` giữ
compiled executor cùng direct-QKV #13, alias `v15` giữ direct-QKV parent,
alias `v14.1` giữ large-sequence rollback, còn `v11` giữ arithmetic rollback.
Correctness gate trước trên
#7/#10/#6, rồi paired V8.1/V11. Đảo thứ tự hai implementations cho direction
thứ hai:

```bash
TORCH_LOGS=perf_hints CUDA_VISIBLE_DEVICES=1 python matrix_runner.py \
  --impl main --shape-ids 1,2,3,4,5,6,7,8,9,10,11,12,13 \
  --accuracy-trials 5 --warmup 1 --repeats 1 --benchmark-rounds 1 \
  --compile-user --compile-mode max-autotune

TORCH_LOGS=perf_hints CUDA_VISIBLE_DEVICES=1 python profile_models.py \
  --impl v8.1 v11 --shape-id 7 --accuracy-trials 10 \
  --warmup 50 --repeats 200 --benchmark-rounds 5 \
  --compile-user --compile-mode max-autotune

TORCH_LOGS=perf_hints CUDA_VISIBLE_DEVICES=1 python profile_models.py \
  --impl v8.1 v11 --shape-id 6 --accuracy-trials 5 \
  --warmup 10 --repeats 20 --benchmark-rounds 3 \
  --compile-user --compile-mode max-autotune
```

### 9.19 Dùng V4.1 standalone

```python
import torch
from v4_1_clean import TransformerConfig, UserOptimizedTransformer

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
model = torch.compile(model, mode="reduce-overhead")
output = model(x, valid_token_mask)
```

File clean không tự compile và không có benchmark CLI; compile phải xảy ra sau
weight loading, device transfer và `eval()` như ví dụ trên.

### 9.20 Dùng V4.3 Flash-first standalone

```python
import torch
from v4_3_flash_clean import TransformerConfig, UserOptimizedTransformer

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
model = torch.compile(model, mode="max-autotune")
output = model(x, valid_token_mask)
```

Fast path causal giả định `valid_token_mask` là right padding dạng prefix
`True` rồi suffix `False`. File không tự compile, không có benchmark CLI và
không import `torch_transformer_benchmark`.

### 9.21 Tái lập V13 INT8 accuracy probe

Official shape #2 W8A8 canary:

```bash
TECHJAM_INT8_PROBE_MODE=w8a8 python3 v13_INT8FFNProbe.py \
  --device cpu --dtype float32 \
  --batch-size 1 --seq-len 128 --d-model 128 \
  --heads 4 --ffn-dim 128 --layers 4 --causal \
  --accuracy-trials 5 --warmup 0 --repeats 1 --benchmark-rounds 1
```

Đổi mode sang `w8` hoặc `a8` để chạy controls. Thêm
`TECHJAM_INT8_PROBE_LAYERS=3` để chỉ quantize layer cuối. Mọi mode hiện fail và
harness tự skip benchmark. Không thêm `--benchmark-on-failure`: graph này là
fake-quant accuracy simulation, không phải implementation INT8 để đo latency.

### 9.22 Tái lập V15 direct-layout QKV

Chỉ chạy khi GPU vật lý #1 idle để Triton autotune không cache tile bị
chọn dưới contention. Command accuracy + paired profile chính cho #13:

```bash
TORCH_LOGS=perf_hints CUDA_VISIBLE_DEVICES=1 .venv/bin/python profile_models.py \
  --impl v14.1 v15 --shape-id 13 --accuracy-trials 5 \
  --warmup 50 --repeats 200 --benchmark-rounds 5 \
  --profile-warmup 50 --profile-iterations 20 \
  --compile-user --compile-mode max-autotune
```

Đảo `--impl v15 v14.1` cho order thứ hai. Promotion evidence phải giữ strict
accuracy và whole-model/raw-device gain ở cả hai orders; isolated QKV hoặc
Flash kernel time không thay thế end-to-end result.

### 9.23 Tái lập V16 compiled batch executor

Chạy full strict #14 và sandwich control/candidate bằng command ở §9.3.1,
đổi `--impl v16` thành `--impl v14.1` cho control. Compile/autotune phải xảy ra
trong warmup; không thêm `--disable-inner-compile` cho candidate measurement.
Kiểm tra GPU idle trước từng process và không giữ output 12.207 GiB cũ giữa
repeats. Baseline/speedup vẫn báo N/A.

### 9.24 Tái lập V17 compiled batch-2 ablation

Chạy full gate bằng command §9.3.1 với `--impl v17`. Sau PASS, chạy alternating
processes V16/V17 với cùng command timing:

```bash
CUDA_VISIBLE_DEVICES=0 /venv/main/bin/python shape14_optimized_benchmark.py \
  --device cuda:0 --impl v17 --warmup 1 --repeats 5 \
  --compile-mode max-autotune
```

Đổi `--impl` qua lại `v16`, `v17`, `v16`, `v17`; kiểm tra GPU idle trước mỗi
process. Không promote từ batch-limit-2 canary hay one-repeat diagnostic. V17
hiện là ablation theo D-033; D-038 đổi `main.py` sang V16.1 vì source topology,
không phải vì V17.

### 9.25 Historical reproduction V17-Sage

V17-Sage đã fail strict official #6/#9 và bị reject theo D-040. Các command
dưới đây chỉ để tái lập artifact; require flag ngăn dependency thiếu biến thành
một fallback PASS giả:

```bash
CUDA_VISIBLE_DEVICES=1 .venv/bin/python v17_sage_opcheck.py \
  --device cuda:0 --seq-len 128 --head-dim 32 --exact-prefix 32

TECHJAM_SAGE_REQUIRE=1 CUDA_VISIBLE_DEVICES=1 .venv/bin/python matrix_runner.py \
  --impl v17.sage \
  --shape-ids 1,2,3,4,5,6,7,8,9,10,11,12,13 \
  --device cuda:0 --dtype float32 \
  --accuracy-trials 5 --warmup 20 --repeats 100 --benchmark-rounds 3 \
  --compile-user --compile-mode max-autotune
```

Historical #14 gate dùng memory-bounded accuracy tool riêng và tắt CUDA Graph:

```bash
TECHJAM_SAGE_REQUIRE=1 CUDA_VISIBLE_DEVICES=1 .venv/bin/python \
  shape14_accuracy.py --device cuda:0 --impl v17.sage \
  --batch-limit 1 --query-chunk 256 --compare-token-chunk 2048 \
  --seed 1234 --compile-mode max-autotune-no-cudagraphs
```

Nếu B=1 PASS, tăng `--batch-limit 2`, rồi `32`. Chỉ sau full strict PASS mới
chạy optimized-only timing:

```bash
TECHJAM_SAGE_REQUIRE=1 CUDA_VISIBLE_DEVICES=1 .venv/bin/python \
  shape14_optimized_benchmark.py --device cuda:0 --impl v17.sage \
  --warmup 1 --repeats 5 --compile-mode max-autotune-no-cudagraphs
```

Để audit raw Sage hoặc prefix lớn hơn, đặt
`TECHJAM_SAGE_EXACT_PREFIX=0|16|32|64`; tolerance/comparator không đổi.

### 9.26 V18-Sage direct performance-only probe

Trên Vast RTX 5090 chỉ có GPU 0. Preflight không timing:

```bash
TECHJAM_SAGE_REQUIRE=1 CUDA_VISIBLE_DEVICES=0 python3 v18_sage_opcheck.py \
  --device cuda:0 --seq-len 128 --head-dim 32
```

Hai sample nhẹ trước full run:

```bash
TECHJAM_SAGE_REQUIRE=1 CUDA_VISIBLE_DEVICES=0 python3 matrix_runner.py \
  --impl v18.sage --shape-ids 2,12 --device cuda:0 --dtype float32 \
  --accuracy-trials 1 --warmup 2 --repeats 10 --benchmark-rounds 1 \
  --compile-user --compile-mode max-autotune --benchmark-on-failure \
  --timeout 1800
```

Full #1–#13 performance diagnostic do owner chạy:

```bash
TECHJAM_SAGE_REQUIRE=1 CUDA_VISIBLE_DEVICES=0 python3 matrix_runner.py \
  --impl v18.sage \
  --shape-ids 1,2,3,4,5,6,7,8,9,10,11,12,13 \
  --device cuda:0 --dtype float32 --accuracy-trials 1 \
  --warmup 20 --repeats 100 --benchmark-rounds 3 \
  --compile-user --compile-mode max-autotune --benchmark-on-failure \
  --timeout 3600
```

`--benchmark-on-failure` không thay tolerance hoặc biến FAIL thành PASS; nó chỉ
cho harness tiếp tục timing và mọi row fail vẫn có status `ACCURACY_FAIL`.
Shape #14 dùng optimized-only tool vì original baseline không executable:

```bash
TECHJAM_SAGE_REQUIRE=1 CUDA_VISIBLE_DEVICES=0 python3 \
  shape14_optimized_benchmark.py --device cuda:0 --impl v18.sage \
  --warmup 1 --repeats 5 \
  --compile-mode max-autotune-no-cudagraphs
```

### 9.27 Gate V19 CUDA checkpointed-FP16

Chạy canary accuracy trước, trên GPU idle. Không đặt
`TECHJAM_V19_ALLOW_CUDA_FALLBACK`; build/kernel lỗi phải dừng thay vì silent đo
V16.1:

```bash
for checkpoint in 16 32 64 128 fp32; do
  TECHJAM_V19_CHECKPOINT_K="$checkpoint" CUDA_VISIBLE_DEVICES=1 \
    python3 matrix_runner.py --impl v19 --shape-ids 7,10,2 \
    --device cuda:0 --dtype float32 --accuracy-trials 5 \
    --warmup 1 --repeats 1 --benchmark-rounds 1 \
    --compile-user --compile-mode max-autotune --timeout 1800
done
```

Chỉ mode PASS mới được chạy full #1–#13 rồi paired V16.1/V19. Ví dụ default
K=32:

```bash
TECHJAM_V19_CHECKPOINT_K=32 CUDA_VISIBLE_DEVICES=1 python3 matrix_runner.py \
  --impl v19 --shape-ids 1,2,3,4,5,6,7,8,9,10,11,12,13 \
  --device cuda:0 --dtype float32 --accuracy-trials 5 \
  --warmup 20 --repeats 100 --benchmark-rounds 3 \
  --compile-user --compile-mode max-autotune --timeout 1800

TECHJAM_V19_CHECKPOINT_K=32 CUDA_VISIBLE_DEVICES=1 python3 profile_models.py \
  --impl main v19 --shape-id 6 --accuracy-trials 5 \
  --warmup 10 --repeats 20 --benchmark-rounds 3 \
  --compile-user --compile-mode max-autotune
```

Đảo `--impl v19 main` cho order thứ hai. Sau #1–#13 PASS, dùng
`shape14_accuracy.py --impl v19` theo B=1, B=2 rồi B=32 trước
`shape14_optimized_benchmark.py`; baseline/speedup #14 vẫn N/A.

### 9.28 Gate V19.1 parallel batch scheduler

Đầu tiên chạy B=2 để buộc outer multi-stream path mà vẫn giới hạn memory:

```bash
TECHJAM_V19_PARALLEL_PARTS=2 CUDA_VISIBLE_DEVICES=1 python3 \
  shape14_accuracy.py --device cuda:0 --impl v19.1.0 --batch-limit 2 \
  --query-chunk 256 --compare-token-chunk 2048 --seed 1234 \
  --compile-mode max-autotune

TECHJAM_V19_CHECKPOINT_K=32 TECHJAM_V19_PARALLEL_PARTS=2 \
  CUDA_VISIBLE_DEVICES=1 python3 shape14_accuracy.py --device cuda:0 \
  --impl v19.1.1 --batch-limit 2 --query-chunk 256 \
  --compare-token-chunk 2048 --seed 1234 --compile-mode max-autotune
```

Nếu strict PASS và còn memory headroom, tăng `--batch-limit` tới 32 trước, rồi
mới thử parts 4/8/16/32 từng mức một. Parts>1 tự ép inner mode thành
`max-autotune-no-cudagraphs`; tool in effective mode và peak allocation để
audit. Dừng sweep khi OOM hoặc headroom không an toàn.

Sau full B=32 strict PASS, đo optimized-only trên GPU idle:

```bash
TECHJAM_V19_PARALLEL_PARTS=2 CUDA_VISIBLE_DEVICES=1 python3 \
  shape14_optimized_benchmark.py --device cuda:0 --impl v19.1.0 \
  --warmup 1 --repeats 5 --compile-mode max-autotune

TECHJAM_V19_CHECKPOINT_K=32 TECHJAM_V19_PARALLEL_PARTS=2 \
  CUDA_VISIBLE_DEVICES=1 python3 shape14_optimized_benchmark.py \
  --device cuda:0 --impl v19.1.1 --warmup 1 --repeats 5 \
  --compile-mode max-autotune
```

Đối chứng tương ứng là V16.1 cho V19.1.0 và V19 K=32 cho V19.1.1, cùng
no-CUDA-Graph policy. Shape #14 vẫn không có executable baseline/speedup chính
thức; báo latency optimized-only và peak memory.

## 10. Mỗi version đóng góp gì

| Chi phí | Baseline | V1 | V2 | V3 | V3.1 |
|---|:---:|:---:|:---:|:---:|:---:|
| Ba Q/K/V GEMM riêng | Có | Không | Không | Không | Không |
| Ba explicit contiguous head copies | Có | Có | Có | Không | Không |
| Explicit attention score/softmax path | Có | Có | Không | Không | Không |
| Materialize causal mask theo `B×S²` | Có | Có | Có | Có khi mask được truyền | Không |
| Attention-level padding zero | Có | Có | Có | Có | Không |
| Attention module dispatch mỗi layer | Có | Có | Có | Không | Không |
| FP16/BF16 behavior | Reference | Packed QKV | V1 math | Full reference fallback | Full reference fallback |

Trên official shape #1, V2 đạt `1.355x`, V3 đạt `1.926x` trong lượt ablation mới và V3.1 đạt `2.152x`; official result của V1 còn pending. Các số `1.035x`, `1.384x` và `1.781x` từ default non-causal shape chỉ là historical diagnostics.

Whole-model `torch.compile(mode="reduce-overhead")` là execution configuration đặt trên V3.1, không phải một implementation file mới và không thay weight/state semantics. Trong paired official-shape #1 run với measurement dài hơn, nó tăng V3.1 từ `2.112x` eager lên `3.404x`; mode attribution và full matrix còn pending.

## 11. Công cụ và quy trình phát triển

- PyTorch được dùng cho reference, optimized implementation, correctness và CUDA Event timing.
- `profile_models.py` dùng PyTorch Profiler/Kineto: eager tổng hợp ATen category/model stage; compiled path tổng hợp raw GPU device/kernel events, Triton/compiled-region/CUDA-Graph launch evidence, top device events và steady/peak CUDA allocation.
- OpenAI Codex được dùng để đối chiếu đề với benchmark, phân tích code, đề xuất và triển khai candidate, chạy test local/GPU và thực hiện ablation giữa các biến thể.
- Input benchmark là tensor sinh ngẫu nhiên với seed cố định; project không dùng dataset bên ngoài.
- Mỗi thay đổi được kiểm tra syntax và accuracy trước khi benchmark hiệu năng.
- Baseline và optimized luôn nhận cùng weights và cùng input trong một process.

## 12. Giới hạn hiện tại

1. V16.1 main bao phủ #1–#14 bằng V14.1/V11 packed-QKV và standalone compiled
   executor. Fresh standalone run đã PASS full #1–#13 và full `B=32` #14;
   multi-seed/input-scale/padding vẫn pending như robustness mở rộng.
2. V3.1 chỉ tối ưu FP32 inference; V4/V4.1 nhận FP32 public input nhưng dùng FP16 internal compute; training và input dtype khác FP32 vẫn fallback.
3. Final #1–#13 JSON/CSV đã đầy đủ và được track. #14 vẫn cần harness riêng vì
   reference score ~18.6 TiB và repeated outputs có lifecycle OOM risk;
   baseline latency/speedup không tồn tại.
4. Đã có eager PyTorch Profiler/Kineto breakdown và compiled GPU event evidence; default-vs-reduce attribution, exported trace review, baseline breakdown, Nsight Systems/Compute, occupancy và memory-traffic evidence còn pending.
5. V4.3 Flash-first đã đo trên RTX 5090; causal mask-elision giả định valid-token mask là prefix-true/suffix-false, không phải arbitrary sparse mask.
6. V7a đã thử residual + LayerNorm pipeline nhưng Inductor sinh cùng fused
   graph với V4.3; chưa có lý do viết custom residual/LayerNorm kernel riêng.
7. V5 per-tensor/MXFP8 chạy được trên hardware nhưng đều FAIL strict model
   accuracy; chưa có calibration/QAT và không có FP8 performance claim hợp lệ.
8. V6 approximate GELU PASS #1–#13 nhưng clean paired #8 chậm hơn `0.32%`, còn
   #2 cho host/device signals trái nhau và thêm một kernel; không promote.
9. V7a residual/LayerNorm PASS targeted accuracy nhưng compiled kernel graph
   trùng V4.3; không promote và không có full #1–#13 matrix.
10. V8.1 force-all có aggregate gain khoảng 1% nhưng #2/#12 order-sensitive và
    #11 raw GPU regression; per-shape dispatcher mở rộng vẫn chưa chốt.
11. V8/V8.1/V14.1/V15/V16/V16.1/V17/V18 chưa được validate trên GPU architecture khác; cutoff `8192`
    là policy cho matrix hiện tại, chưa phải universal optimum.
12. V9 persistent MLP chỉ support D/FFN≤128 và whole-model chưa có stable win;
    isolated kernel speedup không đại diện graph-level latency.
13. V16.1 main dùng packed-QKV V11 cho #1–#13 và compiled chunked path cho #14;
    V16 là rollback có direct-QKV #13. #7 giảm
    max error với host latency regress `1.47–2.77%`; đây là accepted accuracy
    trade-off, không phải Pareto-win.
14. Packed cache cần refresh nếu weights bị sửa trực tiếp trong eval mode.
15. Final CPU/RAM/disk/GPU/software metadata đã lưu trong
    `results/final/environment.json`; portability sang environment khác chưa được chứng minh.
16. V13 symmetric INT8 FFN-in fail official #2 ngay cả ở W8-only một layer;
    chưa có outlier routing, error correction hoặc calibrated/QAT recipe.
17. V15 gain #13 chỉ `0.98–2.20%` trên RTX 5090; schedule và autotune choice
    chưa portable tới GPU khác, còn compile/cold-start cost chưa được tính.
18. V16 predecessor gain #14 `3.11–3.61%` so V14.1 vẫn là historical paired
    control. V16.1 standalone đã PASS full #14 và có optimized-only median
    `6987.4644 ms` trên PyTorch 2.11/cu128/driver 595, nhưng original baseline latency vẫn
    không đo được.
19. V17 B=2 numerical/memory gates PASS nhưng gain chỉ `0.30–0.59%` so V16;
    chưa có profiler attribution hoặc confidence đủ để promote hay thử B=4.
20. Historical V18 SageAttention exact #14 fail strict eager `1/102.4M`;
    corrected V17-Sage tiếp tục fail official #6/#9 dù #13 đạt `17.067x`.
    V18-Sage direct automatic chủ đích bỏ correction và chỉ có giá trị như
    performance diagnostic; optional dependency chưa thuộc submission. FA4 b28
    cũng chậm hơn PyTorch Flash `7.72%` trên isolated exact attention.

## 13. Hướng phát triển tiếp theo

Ưu tiên theo thứ tự:

1. **Probe exact FlashInfer SM120.** Đây là exact backend quan trọng duy nhất
   trong shortlist chưa đo. Gate attention-only và whole-layer trên #13, sau đó
   #14 B1; đưa adapter/layout copy vào timing và giữ PyTorch Flash làm control.
2. **Fuse QKV/layout/memory path.** Thử TE `LayerNormLinear` hoặc CUTLASS/custom
   direct-layout epilogue, rồi nối sang backend-native Q/K/V. Với #14, giữ packed
   weight nhưng thử separate/no-concat activation và scratch reuse để giảm live
   memory; gate từng fusion riêng để còn attribution.
3. **Xây accuracy-aware workload router.** Candidate đầu là direct-layout QKV
   cho large `B*S`, `D=FFN=128`, dựa trên measured win #6/#13. Predicate phải
   dựa trên workload/GPU thay vì official test ID; promotion cần strict full
   matrix, reverse-order/raw-device evidence và aggregate không regress.
4. **Mở rộng robustness và portability gate.** Chạy nhiều seed, input scale,
   padding ratio, causal/non-causal và mask modes; log actual SDPA backend, peak
   allocated/reserved, compile cold-start và steady-state riêng. Validate thêm
   GPU/software stack thứ hai trước khi coi compile/backend policy là portable.
5. **Để exact FFN và small-shape specialization sau đúng profiler.** Chỉ thử
   cuBLASLt/TE/CUTLASS exact GELU trên #6/#8 hoặc `D=32`/`Dh=8` persistent kernel
   trên #7/#11 khi target profile xác nhận bottleneck. Không viết standalone
   residual/LayerNorm Triton khi Inductor đã sinh graph tương đương, và không
   promote persistent MLP từ isolated win.
6. **Giữ low precision ở nhánh deferred/high-risk.** Current BF16/FP8/INT8 và
   Sage recipes đã bị correctness gate chặn. Chỉ mở lại recipe khác bản chất có
   smoothing, protected FP32 boundaries hoặc outlier correction; bắt đầu bằng
   accuracy-only #1/#8/#13 và không timing chính thức khi fail.
7. **Chỉ viết custom SM120 attention sau library ceiling.** Nếu FlashInfer cùng
   layout fusion không thắng hoặc đã chạm trần, mới thử exact online softmax,
   causal triangular load balancing, `S=100000` scheduler hoặc `Dh=8` kernel.
8. **Dừng batch/FFN micro-tuning #14 nếu chưa có evidence mới.** Attention đang
   chiếm `92.258%`; B=2 chỉ thắng `0.30–0.59%`. V19 checkpointed-FP16 và V19.1
   multi-stream tiếp tục là deferred prototypes, không đứng trước attention work.
9. Giữ V16.1 làm main; V16/V17/V15/V14.1/V11 và các candidate khác tiếp tục là
   rollback/ablation. Mọi promotion vẫn phải giữ public API/state dict, PASS
   strict comparator và thắng whole-model trong cùng environment/protocol.

## 14. Kết luận

Artifact nộp cuối là `main.py → v16_1_clean.py`. Fresh official run trên commit
`4f77a04` đã PASS strict cả 14 shapes. Promoted driver-595 start-control
#1–#13 đạt geomean **11.803x**, từ `2.379x` ở #8 tới `38.762x` ở #13, với
worst max abs `0.00179085` và zero
failed elements. Full #14 PASS `0/3.2768B`, optimized-only median
`6987.4644 ms`, throughput `457,963 token/s`, peak `24.487 GiB`; baseline và
speedup giữ N/A vì reference score tensor khoảng `18.6 TiB`. Đây là headline
submission result; driver-580 `7.904x` là cross-host archive và không được diễn
giải thành code delta. Các số V4.3/V15/V16 bên dưới là ablation hoặc predecessor
evidence, không được gán cho artifact cuối.

Packed QKV đơn lẻ là optimization an toàn nhưng official performance còn pending. V2 cho thấy SDPA tạo bước tăng rõ ràng; v3 giải quyết thêm data movement và per-layer dispatch. V3.1 tiếp tục loại causal-mask materialization và padding zero dư. Trên RTX 5090, v3.1 đạt **2.152x** ở official shape #1 và **7.697x** ở shape #13, đồng thời pass strict correctness trên các nhánh đã kiểm tra.

V3.1 vẫn là FP32 implementation ổn định để tiếp tục full-matrix validation. Với
mixed precision, V4.1 bỏ tám copy kernels và cải thiện V4 eager `5.9%`, nhưng
Inductor đã tự fuse cùng conversion nên V4/V4.1 compiled đều đạt **0.1858 ms**
trên official shape #1, nhanh hơn V3.1 compiled `41.4%` trong cùng run. V4.2
dispatch cuDNN theo shape và đạt geomean `7.58x`; `v4_3_Flash.py` tiếp tục bỏ key
mask dư trên causal/right-padding shapes và ưu tiên Flash với backend fallback.
#1–#13 đều PASS; với max-autotune geomean đạt **9.53x** và shape #13 đạt
**37.649x** (`1.1096 ms`). Đây là steady-state fallback nhanh nhất hiện tại cho
đa số shape. V14 bổ sung exact batch-chunk path cho #14 và PASS strict
`0/3.2768B`; V14.1 sau đó đưa cùng algorithm qua cutoff `S >= 8192`, giữ
V11 nguyên cho #1–#13. V15 bọc V14.1 và thay QKV riêng exact #13 bằng
direct-layout Triton; paired dài giảm latency `0.98–2.20%`. V16 tiếp tục giữ
loop #14 eager nhưng compile/reuse thân B=1, full strict PASS và giảm
optimized-only median `3.11–3.61%`, nên được promote qua `main.py` theo D-032.
V17 thử tăng compiled executor chunk lên B=2 và vẫn PASS full strict; alternating
controls cho gain `0.30–0.59%` (trung bình `0.515%`) nhưng peak không đổi và
effect quá nhỏ để promotion, nên D-033 giữ V16 main ở thời điểm đó và V17 làm
ablation. D-038 sau đó promote source-clean V16.1: giữ standalone compiled
executor #14 nhưng bỏ V15/exact-#13 branch, chấp nhận trả lại win
`0.98–2.20%` ở #13 để có source topology không hard-code official tuple.
Inner profiler sau đó cho thấy attention chiếm `92.258%`, chốt đúng bottleneck
thay vì tiếp tục batch/FFN micro-tuning. Built-in Flash vẫn thắng cuDNN,
Efficient và FA4 b28. Sage PV-FP16 nhanh `1.3933x` ở isolated attention nhưng
V18 full-model B=1 còn fail strict một phần tử, đồng thời compiled wrapper chưa
equivalent eager; D-034 vì thế giữ PyTorch Flash và chặn mọi performance claim
cho V18. Backend này tiếp tục được V16.1 main sử dụng.
Robustness nhiều seed/hardware và scorer lifecycle dài vẫn là follow-up, nhưng
official single-seed final gate đã đóng. V1, v2, v3, V4.1,
V4.2 và forced-backend files được giữ làm ablation để giải thích nguồn speedup.
V5 xác nhận RTX 5090 chạy được per-tensor FP8 và native MXFP8, nhưng full lẫn
single-projection variants đều vượt error budget; chúng được giữ như negative
ablation và bị accuracy gate chặn trước performance benchmark. V5.1 full FP16
accumulation cũng bị loại: reduce-overhead fail #8, còn max-autotune vừa fail
#10 vừa không nhanh hơn V4.3 trong paired #8. V6 tanh GELU PASS accuracy
#1–#13 nhưng clean paired #8 regression `0.32%`; #2 cho host/device signals trái
nhau và làm mất một epilogue fusion, nên cũng không thay V4.3.
V7a pipeline residual/LayerNorm cũng PASS targeted correctness nhưng sinh đúng
cùng max-autotune kernel graph; paired #2/#8/#12 không có gain ổn định. Vì vậy
custom V7b bị dừng trước khi thêm complexity. V8 sau đó nhắm đúng profiler gap
FFN: custom Triton FFN-in GEMM + exact-GELU giảm official #6 từ
`26.6799 → 25.4092 ms` (`-4.76%`) và `32 → 29` kernels, đồng thời full
#1–#13 accuracy PASS. #8 không có end-to-end gain dù isolated kernel nhanh hơn,
nên V8 chỉ dispatch cho `B*S >= 1M, D=FFN=128` và giữ V4.3 làm fallback. Đây là
best path hiện tại cho #6. V8.1 force-all sau đó xác nhận full #1–#13 accuracy
PASS và aggregate latency giảm `0.97–1.36%` so với V4.3 theo hai orders, nhưng
#2/#12 order-sensitive và #11 raw GPU regression. Do đó unconditional fusion
không được promote; evidence mới chỉ biện minh một dispatcher whitelist tinh
hơn. V9 fully fused persistent MLP tiếp tục bỏ cả hidden materialization và
FFN-out launch: isolated FFN nhanh `1.18–1.59x`, nhưng whole-model #1/#5/#13
regress và #7 đổi dấu theo order; #6 cũng chậm hơn V8 `0.16%`. Do đó V9 được
giữ làm ablation, còn V8 vẫn là best path #6. Portability qua GPU khác vẫn
pending. V11 bỏ riêng FP16 round trước exact GELU và PASS GPU #1–#13.
So với V8.1, max abs giảm `13.01%` ở #6 với latency trung tính, còn #10 vừa giảm
max abs `1.19%` vừa nhanh hơn khoảng `1.50%`; #7 giảm max abs `4.85%` nhưng host
latency regress `1.47–2.77%`. Project owner chấp nhận trade-off này và promote
force-all V11; V14.1 bọc arithmetic này thành large-sequence parent, V15 thêm
direct-QKV #13, V16 thêm compiled #14 trên topology đó, còn V16.1 là current
source-clean main wrapper. V8/V8.1/V11/V14.1/V15/V16 và V17 được giữ làm
controls/rollback/ablation.
V13 sau đó
thử symmetric INT8 riêng FFN-in nhưng official #2 fail cả W8, A8 và W8A8; ngay
cả W8-only layer cuối cũng fail strict comparator. Accuracy gate vì thế chặn
GPU/kernel work và không có INT8 performance claim. V14/V16 predecessor đã mở
đường cho #14; fresh V16.1 standalone run nay đã đóng full `B=32` accuracy và
optimized-only timing. Robustness matrix còn mở, còn paired #14 speedup không
có vì original reference không executable trên 32 GiB.
