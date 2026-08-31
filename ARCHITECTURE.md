# Kiến trúc repository

## 1. Phạm vi

Repository xây dựng một harness để so sánh Transformer reference với implementation tối ưu. Mục tiêu là giảm inference latency trên GPU mà không vượt ngưỡng sai số của đề bài.

Public repository: [wheres-my-perry/techjam-2026-track3](https://github.com/wheres-my-perry/techjam-2026-track3).

## 2. Bản đồ repository

| Thành phần | Vai trò |
|---|---|
| `STATEMENT.md` | Bản lưu đề Track 3, tài nguyên, deliverables và tiêu chí chấm điểm. |
| `torch_transformer_benchmark.py` | Reference implementation và benchmark harness gốc. |
| `main.py` | Benchmark entrypoint trỏ tới standalone `v16_1_clean.py`. |
| `v16_1_clean.py` | Active V16.1: model/config/kernel/cache/executor đầy đủ, không import harness hoặc version cũ. |
| `archive/versions/` | Toàn bộ implementation `v1`–`v18` lịch sử, giữ làm evidence/rollback nhưng không còn là runner target. |
| `v1_fuseQKV.py` | V1: gộp Q/K/V projection thành một phép `F.linear`. |
| `v2_SPDA.py` | V2: V1 + PyTorch SDPA. |
| `v3_SDPA_NoCopy.py` | V3: packed-QKV no-copy + SDPA + flattened model loop. |
| `v3_1_CausalMask.py` | Candidate FP32 hiện tại: v3 không materialize causal mask và không zero-out padding dư trong attention. |
| `v4_FP16.py` | V4: GEMM/SDPA FP16 nội bộ, norm/residual/output FP32. |
| `v4_1_FP16_GELU.py` | V4.1: GELU chạy trực tiếp FP16. |
| `v4_2_SDPA_Dispatch.py` | V4.2: chọn cuDNN/automatic SDPA theo shape. |
| `v4_3_Flash.py` | Candidate V4.3 cuối: causal/right-padding bỏ key mask, ưu tiên Flash với cuDNN/Efficient/Math fallback. |
| `v4_3_flash_clean.py` | Bản V4.3 standalone chỉ chứa config/model Flash-first; không phụ thuộc benchmark harness và không phải runner target. |
| `v5_1_FP16Accum.py` | V5.1 negative ablation: V4.3 + CUDA full FP16 GEMM accumulation; không promote vì accuracy/performance gate. |
| `v6_ApproxGELU.py` | V6 ablation: V4.3 + tanh-approximated FP16 GELU; accuracy PASS #1–#13 nhưng chưa có measurable gain. |
| `v7_ResidualLayerNorm.py` | V7a ablation: pipeline residual + LayerNorm bằng pure PyTorch; Inductor sinh cùng graph V4.3 nên không promote. |
| `v8_FusedFFNGELU.py` | V8: V4.3 + Triton FFN-in GEMM/bias/exact-GELU cho workload D=FFN=128 từ một triệu token; shape khác fallback V4.3. |
| `v8_1_FusedFFNGELUAll.py` | V8.1 ablation: ép V8 custom FFN/GELU trên mọi mixed-precision shape để đo dispatcher; không promote unconditional. |
| `v9_PersistentMLP.py` | V9 ablation: fully fused persistent FFN-in/exact-GELU/FFN-out cho D/FFN≤128; isolated win nhưng whole-model không promote. |
| `v11_FP32PreGELU.py` | Arithmetic parent/rollback: V8.1 nhưng exact GELU đọc trực tiếp FP32 FFN-in accumulator; GPU #1–#13 PASS. |
| `v12_FP32FFNOut.py` | V12 ablation: V11 nhưng FFN-out GEMM store trực tiếp FP32 để bỏ output round FP16; local gate PASS, GPU pending. |
| `v12_1_FP32OutProj.py` | V12.1 ablation: chỉ attention out-projection store trực tiếp FP32; local gate PASS, GPU pending. |
| `v12_2_FP32ResidualOutputs.py` | V12.2 ablation: cả attention-out và FFN-out store trực tiếp FP32; local gate PASS, GPU pending. |
| `v13_INT8FFNProbe.py` | V13 accuracy-only ablation: symmetric INT8 FFN-in với per-channel weight/per-token activation scales; official #2 FAIL nên không có kernel/performance path. |
| `v14_BatchChunked.py` | V14 validated candidate: exact shape-#14 batch chunking trên V11. |
| `v14_1_BatchChunked.py` | Memory parent/rollback: V11 dưới cutoff; FP32 eval với `S >= 8192`, `B > 1` chạy batch chunk size 1. |
| `v15_DirectQKVLayout.py` | QKV parent/rollback: exact #13 dùng direct-layout QKV Triton; mọi branch khác kế thừa V14.1. |
| `v15_1_DirectQKVAll.py` | Cross-shape ablation: force direct-layout QKV cho causal `S<8192`; #1–#12 PASS nhưng chỉ #6 thắng ổn định, không phải main. |
| `v16_CompiledBatchExecutor.py` | Previous main/QKV rollback: giữ loop #14 eager, compile/reuse thân B=1 và kế thừa direct-QKV #13 từ V15. |
| `v16_1_NoDirectQKV13.py` | Historical composed V16.1, nay nằm trong `archive/versions/`. |
| `v17_CompiledBatch2.py` | Experimental: cùng compiled large-sequence executor nhưng batch chunk 2; strict PASS, không promote vì gain chỉ khoảng 0.5%. |
| `v17_sage.py` | V17-Sage negative ablation: corrected Sage trên V16.1; full #1–#13 fail strict ở #6/#9 nên không promote. |
| `v18_sage.py` | V18-Sage performance-only ablation: V16.1 + direct automatic Sage SM120; không correction, #8 fallback vì Dh=256. |
| `v4_mixed_precision_common.py` | Dependency nội bộ của `v4_3_Flash.py`; giữ cache/forward mixed precision. |
| `v4_1_clean.py` | Bản V4.1 standalone chỉ chứa config/model FP16; không phụ thuộc benchmark harness và không phải runner target. |
| `matrix_runner.py` | Chạy một implementation trên đúng 14 official shapes và xuất JSON/CSV. |
| `profile_models.py` | Accuracy gate, benchmark và PyTorch Profiler/Kineto cho nhiều implementation trên cùng official shape. |
| `AGENTS.md` | Quy tắc làm việc cho người và coding agent. |
| `SOLUTION.md` | Technical report đầy đủ cho các implementation hiện tại. |
| `EXPERIMENTS.md` | Nhật ký phương án, thử nghiệm và kết quả. |
| `IMPLEMENTATION_PLAN.md` | Roadmap, phase và trạng thái triển khai. |
| `DECISION.md` | Nhật ký quyết định kỹ thuật dài hạn. |
| `tmp/` | Tài liệu/thành phẩm tạm; không thuộc runtime benchmark. |

## 3. Mô hình Transformer reference

`BaselineTransformer` chứa nhiều `BaselineTransformerBlock`, sau đó là final LayerNorm.

Mỗi block chạy theo thứ tự:

1. Pre-LayerNorm.
2. Multi-head self-attention.
3. Residual add.
4. Pre-LayerNorm.
5. FFN: Linear → GELU → Linear.
6. Residual add.
7. Zero-out các query position bị padding, nếu có mask.

Attention reference thực hiện ba projection Q/K/V riêng, tạo score `QKᵀ / sqrt(d_k)`, áp causal/key mask, softmax ở FP32, nhân với V rồi qua output projection.

## 4. Luồng thực thi benchmark

```text
CLI arguments
    ↓
TransformerConfig + argument validation
    ↓
Construct baseline and optimized models
    ↓
Copy identical model weights
    ↓
Move both models to the same device/dtype and switch to eval
    ↓
Optional torch.compile
    ↓
Accuracy trials on generated inputs
    ↓ pass
Warmup both implementations
    ↓
Alternating benchmark rounds
    ↓
Latency statistics, throughput and median speedup
```

Nếu accuracy fail, benchmark mặc định dừng. `--benchmark-on-failure` chỉ dùng để điều tra và không tạo kết quả hợp lệ để báo cáo.

`matrix_runner.py` gọi harness trong một subprocess riêng cho mỗi official shape. Cách ly process giúp runner tiếp tục sau accuracy failure, OOM, process crash hoặc timeout; JSON/CSV được ghi lại sau từng case để giữ partial results. Bảng tổng kết terminal hiển thị `max_abs` cạnh status và latency để nhìn nhanh error margin; quyết định PASS/FAIL vẫn dựa trên comparator strict OR và số phần tử fail.

`profile_models.py` chạy mỗi implementation trong subprocess riêng để tránh import/global-state nhiễu giữa các version. Mỗi child chạy accuracy gate, đo baseline/optimized bằng cùng workload và CUDA Event, sau đó profile optimized path bằng PyTorch Profiler/Kineto. CLI hỗ trợ `--compile-baseline`, `--compile-user` và `--compile-mode` sau bước copy weight/device/eval giống benchmark harness.

Profiler dùng hai schema attribution có chủ đích:

- **Eager:** profiler-only scopes tách pre-attention norm, packed/separate QKV projection, view/reshape, fused attention core, output projection, residual, pre-FFN norm, từng FFN projection, GELU, masking/copy và final norm. Các scope này không đi vào benchmark path.
- **Compiled:** không monkey-patch ATen/stage scopes sau khi graph đã compile vì thao tác đó có thể làm graph recompile hoặc thay đổi fusion. Thay vào đó, tool tổng hợp raw GPU device events, kernel/memory-event count, Triton event/launch count, `Torch-Compiled Region`, CUDA Graph launch API, steady/peak allocation và top device-event names. `TORCHINDUCTOR_UNIQUE_KERNEL_NAMES=1` được bật trong child để trace dễ audit.

Terminal hiển thị latency cùng GPU/runtime evidence cho cả hai path; bảng ATen category/model stage chỉ xuất cho eager. JSON được lưu vào `profile-results/`, còn Chrome trace chỉ sinh khi bật `--export-traces`.

## 5. Dữ liệu và mask

- Input shape: `[B, S, D]`.
- Attention heads: `[B, H, S, D/H]`.
- `valid_token_mask`: boolean tensor `[B, S]`.
- Key padding mask broadcast thành `[B, 1, 1, S]`.
- Causal mask có shape `[S, S]` và chỉ cho phép attention tới token hiện tại hoặc trước đó.
- Output padding position được đưa về 0 để giữ hành vi reference.

Input được sinh bằng seed cố định. Accuracy dùng nhiều trial; performance dùng một input cố định để loại thời gian sinh dữ liệu khỏi latency.

## 6. Đo correctness

Mỗi phần tử pass nếu thỏa ít nhất một điều kiện:

```text
abs(user - reference) < atol
abs(user - reference) < rtol * abs(reference)
```

Hai điều kiện được kết hợp bằng OR. Harness còn báo max absolute error, max relative error, số phần tử fail và vị trí worst case. Repository chốt theo trang đề bài: strict `<`, `atol=0.002`, `rtol=0.02`. Attachment ngày 2026-08-27 dùng `<=`; khác biệt này được ghi trong `DECISION.md`.

## 7. Đo hiệu năng

- CUDA dùng `torch.cuda.Event` trên current stream và synchronize trước/sau batch đo.
- CPU dùng `time.perf_counter_ns()`.
- Cả hai model đều warm up trước khi đo.
- Thứ tự baseline/optimized được đảo theo round để giảm bias do nhiệt độ và clock.
- Kết quả gồm median, mean, p90, min, throughput và speedup theo median.

Median latency là số chính để so sánh; p90 giúp phát hiện variance. Không so sánh các lần chạy khác device, dtype hoặc cấu hình compile như thể chúng cùng một thí nghiệm.

Performance report chỉ nhận shape khớp chính xác một trong 14 test shapes ở Appendix của `STATEMENT.md`. Default shape và các biến thể non-causal/padding tự tạo vẫn hữu ích cho correctness hoặc ablation, nhưng không được dùng làm benchmark chính thức.

## 8. Kiến trúc lịch sử `v1_fuseQKV.py`

Main storyline V1 → V4.3 được giữ trong `archive/versions/`. Các forced-backend,
static-dispatch phụ và V5 negative experiment đã bị xóa trước đó; kết quả của
chúng vẫn nằm trong experiment log.

`v1_fuseQKV.py` giữ nguyên cấu trúc parameter của baseline và thêm cache không persistent:

- Gộp Q/K/V weight và bias thành một packed projection.
- Refresh packed weights sau `load_state_dict()`.
- Chạy một `F.linear` rồi tách kết quả thành Q, K, V.
- Giữ nguyên toàn bộ attention math, mask và FFN của baseline.
- Khi training, dùng lại ba projection gốc để tránh cache stale.

Repository hiện có V4.3 Flash-first SDPA candidate với backend fallback; chưa có custom CUDA/Triton kernel hoặc hardware-aware registry tổng quát.

## 9. Kiến trúc `v2_SPDA.py`

`v2_SPDA.py` kế thừa đúng cấu trúc của v1: một packed QKV projection, ba lần split-head `.contiguous()` và attention module dispatch ở từng layer. Thay đổi duy nhất trên FP32 path là dùng `F.scaled_dot_product_attention` thay cho explicit scores/softmax/context math. Causal-only dùng `is_causal`; causal kết hợp padding dùng boolean mask. FP16/BF16 giữ attention math của v1 để bảo toàn correctness.

## 10. Kiến trúc `v3_SDPA_NoCopy.py`

`v3_SDPA_NoCopy.py` là bản packed-QKV + SDPA đã tối ưu thêm data movement và dispatch. Q/K/V được tạo bằng `reshape → permute → unbind` mà không gọi `.contiguous()` ba lần. Attention, mask dùng chung, residual và FFN chạy trong một whole-model loop để bỏ attention module dispatch mỗi layer. Packed weights được refresh sau `load_state_dict()`; training và dtype khác FP32 fallback toàn bộ về reference.

### 10.1 Kiến trúc `v3_1_CausalMask.py`

V3.1 giữ nguyên v3 nhưng truyền key-padding mask `[B,1,1,S]` trực tiếp vào SDPA cùng `is_causal=True`, thay vì materialize mask `[B,1,S,S]`. Invalid query chỉ được zero một lần cuối mỗi block và sau final LayerNorm; không zero riêng attention output vì giá trị này không thể ảnh hưởng token hợp lệ trước điểm zero cuối block.

### 10.2 Kiến trúc V4 mixed precision

`v4_FP16.py` và `v4_BF16.py` giữ public parameters/state dict ở FP32 nhưng tạo
non-persistent cache FP16/BF16 cho packed QKV, attention output projection và hai
FFN projection. Pre-LayerNorm, residual add, GELU và final LayerNorm chạy FP32;
SDPA cùng các GEMM chạy internal dtype rồi cast kết quả về FP32 trước residual.
Cache được rebuild sau `load_state_dict()` và `_apply()` để không bị `.to(...,
dtype=float32)` vô tình đổi cache về FP32. Training và input dtype khác FP32 dùng
reference fallback. Hai candidate phải qua correctness gate độc lập trước khi có
kết quả performance hợp lệ.

`v4_1_FP16_GELU.py` dùng cùng core/cache FP16 nhưng đặt GELU trực tiếp trên
hidden FP16. V4 gốc vẫn giữ GELU FP32 để làm control. V4.1 loại hai dtype casts
mỗi layer; LayerNorm, residual accumulation và output vẫn FP32. Candidate này
được version hóa riêng vì thay precision của activation có thể dùng thêm error
budget và phải qua accuracy gate độc lập.

`v4_1_clean.py` mirror cùng parameter names, non-persistent cache lifecycle,
optimized forward và safe fallback của V4.1 trong một module standalone. File
không import `torch_transformer_benchmark`, không có CLI và không chứa baseline;
caller chịu trách nhiệm load weights, chuyển device, gọi `eval()` và bọc
`torch.compile` nếu cần.

### 10.3 Kiến trúc V4.2 SDPA dispatch

Ba forced-backend file Flash/Efficient/cuDNN cô lập đúng backend SDPA để
benchmark. `v4_2_SDPA_Dispatch.py` là candidate kết hợp: config key
`(B,S,D,H,L,FFN,causal)` được tra một lần khi khởi tạo model. Các key đã validate
#1/#2/#3/#4/#7/#9/#13 ép cuDNN; key khác giữ automatic PyTorch dispatch. Vì lựa
chọn là Python constant trước `torch.compile`, hot path vẫn có một compiled
region và một CUDA Graph launch, không có tensor condition hoặc host sync.

### 10.4 Kiến trúc V4.3 Flash-first

V4.3 tận dụng invariant của causal self-attention với right padding: valid tokens
là một prefix, nên valid query không thể nhìn thấy padded keys nằm trong suffix
do causal constraint; invalid query outputs vẫn được zero cuối block. Nhờ đó
causal optimized path có thể truyền `attn_mask=None` và dùng PyTorch Flash SDPA
mà không gọi `.all().item()` hoặc thêm tensor branch trong forward.

`v4_3_Flash.py` dùng một priority list ngắn cho mọi causal shape:
Flash → cuDNN → Efficient → Math. PyTorch bỏ qua backend không hỗ trợ device/shape;
non-causal giữ key mask và automatic dispatch. Training hoặc public input khác
FP32 vẫn về reference path. Fast path yêu cầu valid-token mask dạng
prefix-true/suffix-false; arbitrary sparse masks không dùng được phép chứng minh
causal này.

`v4_3_flash_clean.py` mirror cùng parameter names, strict-compatible state dict,
non-persistent FP16 cache lifecycle, optimized forward và fallback của V4.3
trong một module standalone. File không import `torch_transformer_benchmark`,
không có CLI và không phải target của matrix/profile runner; caller chịu trách
nhiệm load weights, chuyển device, gọi `eval()` và bọc `torch.compile` nếu cần.

### 10.5 V5 FP8 negative ablations

Phần này chỉ giữ lại kết luận lịch sử; toàn bộ source V5 đã được xóa khỏi
working tree sau khi các ablation đều không qua correctness gate.

`v5_FP8.py` dùng E4M3 per-tensor cho packed QKV, attention-output và hai FFN
GEMM; activation scale được tính động còn weight/scale được cache sau load/move.
SDPA vẫn FP16 vì target runtime không hỗ trợ FP8 Q/K/V. Per-tensor model FAIL
strict accuracy trên official #2.

`v5_1_MXFP8.py` thử native Blackwell MXFP8: mỗi block 32 phần tử có E8M0
power-of-two scale, được pad/swizzle sang layout 32×4×4 trước
`F.scaled_mm`. All-ones GEMM sanity PASS, chứng minh kernel/layout đúng, nhưng
full model vẫn FAIL. `v5_2_MXFP8_*.py` hạ từng projection scope riêng để cô lập
error; QKV-only, attention-output-only và từng FFN projection đều FAIL. Các file
này là evidence/ablation, không phải scheduler candidates và không có
performance claim.

### 10.6 V5.1 full FP16 accumulation negative ablation

`v5_1_FP16Accum.py` kế thừa V4.3 và bật
`torch.backends.cuda.matmul.allow_fp16_accumulation=True` trước compile. Flag
backend này là process-global, vì vậy chỉ an toàn trong kiến trúc hiện tại do
matrix/profile runner cách ly mỗi implementation bằng subprocess. Public input,
LayerNorm, residual và output vẫn FP32; chỉ eligible internal FP16 GEMM đổi
accumulation policy.

Max-autotune #1–#13 PASS 12 shapes nhưng shape #10 fail strict comparator một
phần tử qua năm trial. Paired #8 không nhanh hơn V4.3 và có cùng Triton kernel
mix. Reduce-overhead #8 fail `40,029/41,943,040` phần tử. Candidate được giữ để
tái lập kết luận, không nằm trong scheduler hoặc best-path map. Tên MXFP8 cũ
được gọi rõ là `V5.1-MXFP8`; alias runnable `v5.1` dành cho file mới.

### 10.7 V6 approximate GELU ablation

`v6_ApproxGELU.py` kế thừa V4.3 và chỉ override `_mixed_ffn()`: hai Linear,
FP16 cache, residual và public output giữ nguyên; `F.gelu(...,
approximate="none")` được đổi thành `approximate="tanh"`. Training và input
khác FP32 vẫn đi qua reference fallback kế thừa.

Candidate PASS strict accuracy trên official #1–#13. Ở clean paired
max-autotune #2, host median có vẻ giảm (`0.0780 → 0.0749 ms`) nhưng raw device
time tăng (`0.0554 → 0.0560 ms`) và V6 thêm một compiled kernel, nên tín hiệu
không tự nhất quán. Clean paired #8 cho V6 chậm hơn `0.32%`
(`2.5502 → 2.5584 ms`). Full matrix performance bị loại do một training
workload khác chiếm GPU mục tiêu. V6 không nằm trong best-path map.

### 10.8 V7 residual + LayerNorm pipeline ablation

`v7_ResidualLayerNorm.py` giữ attention/FFN branch ở FP16 tới residual boundary
và pipeline mỗi pre-normalization thành cặp output `(residual_fp32,
normalized_fp16)`. Ordering mask, FP32 LayerNorm/residual, exact GELU,
Flash-first SDPA, state dict, cache và fallback giữ nguyên V4.3. Local eager
causal/non-causal × padding/no-padding khớp V4.3 bit-for-bit.

Trên RTX 5090 max-autotune, strict #7/#10 PASS. Paired #2/#8/#12 cho cùng
kernel count, Triton-event count và event names với V4.3. #2 raw GPU time bằng
nhau; #8 V7a chậm `0.32%`; #12 đổi dấu khi đảo thứ tự. Event
`triton_tem_fused__to_copy_addmm_native_layer_norm_t_view_1` cùng các event
fused add/mask/LayerNorm chứng minh Inductor đã thực hiện fusion mục tiêu. V7a
không nằm trong best path; V7b custom Triton không được viết vì không còn
unfused profiler gap và có thể phá GEMM-template fusion hiện tại.

### 10.9 V8 fused FFN-in GEMM + exact GELU epilogue

`v8_FusedFFNGELU.py` kế thừa V4.3 và thay riêng `FFN-in Linear → exact GELU`
bằng một custom Triton kernel. Kernel dot FP16 với accumulator FP32, cộng bias,
round kết quả Linear về FP16 tại đúng precision boundary của V4.3, rồi tính GELU
erf từ giá trị đã round và store FP16 cho FFN-out. `torch.library.custom_op` cùng
fake implementation cho phép `torch.compile` capture kernel; CPU/non-Triton dùng
exact PyTorch fallback.

Isolated #8 microkernel nhanh hơn compiled PyTorch, nhưng end-to-end #8 chỉ dao
động quanh zero vì GEMM template của Inductor hiệu quả hơn khi nằm trong whole
graph. Official #6 (`B*S=1,280,000`, `D=FFN=128`) là case có gain ổn định:
paired max-autotune giảm `26.6799 → 25.4092 ms`, raw GPU time
`26.6680 → 25.3948 ms`, và GPU kernels `32 → 29`. Vì vậy dispatch chỉ bật
custom kernel khi `B*S >= 1,000,000` và `D=FFN=128`; mọi config khác giữ nguyên
V4.3. Full official #1–#13 PASS strict accuracy; shape #14 vẫn pending.

### 10.10 V8.1 force-all dispatcher ablation

`v8_1_FusedFFNGELUAll.py` chỉ kế thừa V8 rồi đặt
`_use_fused_ffn_gelu=True`; không đổi kernel, precision hay model semantics.
Full #1–#13 accuracy PASS. Hai paired directions so với V4.3 cho geomean
latency giảm lần lượt `1.365%` và `0.969%`, đồng thời thường giảm 3–4 kernels.

Gain không đồng đều: #1/#3/#4/#5/#6/#10 giữ dấu nhanh hơn; #2 host latency
regress ở reverse order, #12 đổi dấu hoàn toàn, và #11 raw GPU time chậm hơn
`0.41–0.52%`. Vì vậy V8.1 được giữ làm ablation để thiết kế dispatcher, không
thay V8 stable bằng unconditional custom path.

### 10.11 V9 fully fused persistent MLP ablation

`v9_PersistentMLP.py` fuse hai FFN GEMM và exact GELU vào một Triton custom op.
Mỗi program giữ hidden tile on-chip, loop theo FFN dimension rồi chỉ store final
projection. Hai GEMM accumulate FP32; kernel round FP16 sau FFN-in+bias, sau
GELU và sau FFN-out+bias để giữ precision boundaries V4.3. Support envelope là
D/FFN multiples of 16 tới 128; shape lớn hơn fallback V4.3.

Isolated kernel nhanh hơn compiled two-GEMM reference `1.18–1.59x` trên mọi
unique official FFN shape trong support envelope và full #1–#13 strict accuracy
PASS. Nhưng whole-model #1/#5/#13 regress `4.7%/2.0%/5.3%`; #2/#12 hòa; #7
đổi dấu khi đảo order. #6 V9 `25.4481 ms` gần nhưng chậm hơn V8
`25.4071 ms`. Kernel count `25` thấp hơn V8 `29` và V4.3 `32–33`, song không
chuyển thành score latency. V9 không nằm trong best-path dispatcher.

### 10.12 V11 FP32 pre-GELU accumulator — main arithmetic path

`v11_FP32PreGELU.py` kế thừa V8 và ép custom FFN-in/GELU path như V8.1 để có
đối chứng trực tiếp trên mọi shape. Activation, weight và bias vẫn FP16; dot
product vẫn accumulate FP32; FFN-out, attention, residual/LayerNorm, cache và
fallback không đổi. Điểm khác duy nhất trong Triton epilogue là exact GELU nhận
`accumulator + bias` ở FP32, không round Linear output xuống FP16 rồi cast ngược
lên FP32. GELU output vẫn store FP16 cho Tensor Core FFN-out nên không thêm
kernel, tensor trung gian hoặc output traffic.

Local CPU diagnostics PyTorch 2.12.1 đã PASS strict #7/#10 qua 10 seed, các
nhánh causal/non-causal × padding/no-padding, state-dict, training/FP16/BF16
fallback và `torch.compile` capture. RTX 5090 max-autotune sau đó PASS strict
#1–#13 với worst max abs `0.00179082`. Paired hai order cho #6 giảm max abs
`13.01%` với latency trung tính, #10 giảm max abs `1.19%` và latency khoảng
`1.50%`, còn #7 giảm max abs `4.85%` nhưng host latency regress `1.47–2.77%`.
Theo D-027, force-all V11 được promote dù không Pareto-win ở #7. D-030 sau đó
bọc nguyên V11 trong V14.1 large-sequence dispatcher; D-031 tiếp tục bọc V14.1
bằng V15 direct-QKV cho exact #13. V11 vẫn là arithmetic rollback.

### 10.13 V12 FP32 FFN-out projection output ablation

`v12_FP32FFNOut.py` kế thừa V11 và chỉ đổi precision boundary cuối FFN branch.
Hidden activation, FFN-out weight và cached bias vẫn FP16. Trên CUDA,
`torch.mm(..., out_dtype=torch.float32)` giữ FP32 accumulator khi store output;
cached bias được promote và cộng ở FP32. CPU fallback nhân chính các operand
đã quantize FP16 sau khi promote FP32. Attention output projection, V11
pre-GELU epilogue, residual/LayerNorm, cache/state dict và safe fallback không
đổi.

V12 không giả định bỏ được một cast kernel riêng: V7/profile evidence đã xác
nhận Inductor fuse cast hiện tại vào residual/LayerNorm. Mục tiêu là mua thêm
accuracy margin bằng cách bỏ FP16 output round, đổi lại FP32 intermediate có
thể tăng memory traffic. Local syntax/state-dict, causal/non-causal ×
padding/no-padding, training/FP16/BF16 fallback và compile smoke đều PASS;
paired local 10 seed cho #7/#10 giảm mean error ở 10/10 seed. CUDA accuracy và
paired latency vẫn là gate trước mọi promotion.

V12.1 cô lập attention out-projection bằng cách mirror packed-QKV/SDPA path của
V11 rồi dùng cùng FP32-output helper tại projection cuối. V12.2 kết hợp V12 và
V12.1 để bỏ cả hai residual-boundary rounds. Trên local 10 seed, V12.2 cho mean
error tốt nhất ở #7/#10; V12.1 cho worst max abs tốt nhất ở #10. Cả ba giữ
state dict/fallback và strict branch coverage; GPU mới quyết định trade-off.

### 10.14 V13 INT8 FFN-in accuracy probe

`v13_INT8FFNProbe.py` kế thừa V11 và chỉ thay FFN-in bằng numerical simulation
trước khi có custom kernel. Weight được cache signed INT8 với symmetric
per-output-channel scale; activation sau FP16 LayerNorm boundary dùng dynamic
symmetric per-token scale. W8A8 nhân bằng `torch._int_mm`, accumulate INT32 rồi
dequantize FP32 trước bias và exact GELU; GELU output vẫn FP16 cho FFN-out.

Environment controls `TECHJAM_INT8_PROBE_MODE=w8|a8|w8a8` tách nguồn error và
`TECHJAM_INT8_PROBE_LAYERS=all|0,2,...` cô lập layer scope. Đây không phải
performance graph: quantize/dequantize dùng PyTorch ops để gate số học rẻ trước
Triton/CUTLASS. Official shape #2 fail cả W8, A8 và W8A8 qua năm seed; ngay cả
W8 chỉ ở layer cuối vẫn fail. Vì vậy V13 không nằm trong scheduler, không chạy
GPU performance và không thay promoted main.

### 10.15 V14.1 large-sequence batch dispatcher

`v14_1_BatchChunked.py` kế thừa nguyên V11. FP32 eval với batch lớn hơn một và
runtime sequence length `S >= 8192` được chạy theo batch chunk size 1; mỗi
sample đi qua đủ số layer bằng chính V11 rồi được copy vào output preallocated.
Batch samples không tương tác nên schedule này giữ nguyên QKV, attention, FFN,
residual, LayerNorm và mask semantics.

Cutoff `8192` là boundary bảo thủ: toàn bộ official #1–#13 có `S <= 1024`, còn
#14 có `S=100000`. `B=1`, training và public dtype khác FP32 fallback trực tiếp
V11. Large-sequence helper được đánh dấu `torch.compiler.disable` để Dynamo
không unroll 32 chunk vào một graph lớn; nhánh V11 dưới cutoff vẫn compile bình
thường. D-030 từng promote wrapper này qua `main.py`; D-031 giữ nó nguyên làm
parent/rollback dưới V15, còn V11/V14 vẫn là ablation versioned.

### 10.16 V15 direct-layout QKV dispatcher

`v15_DirectQKVLayout.py` kế thừa nguyên V14.1 và chỉ thay packed QKV projection
cho exact official #13 causal FP32 eval. Custom Triton op nhận activation,
packed weight và bias FP16, accumulate FP32, rồi map store thẳng vào contiguous
`[3,B,H,S,Dh]`. Q/K/V vì vậy có sequence stride `Dh=32` thay cho `3D=384` của
packed view; không có transpose kernel trung gian.

Dispatch key được chốt khi model khởi tạo. Training, non-FP32, non-causal,
shape lạ và #14 đều chạy nguyên V14.1. `torch.library.custom_op` có fake
registration để Inductor capture; CPU/no-Triton fallback materialize cùng
layout bằng PyTorch cho semantic testing.

Theo D-031, V15 được promote sau khi official #13 PASS strict
`0/41,943,040` và paired max-autotune giữ gain ở cả hai orders: end-to-end giảm
`0.98–2.20%`, raw GPU giảm `1.17–1.76%`. Compiled topology vẫn có 29 GPU
kernels nhưng Triton GPU events giảm `21 → 17`; Flash kernel time gần trung
tính, nên improvement được quy cho QKV/projection-layout region.

### 10.17 V16 reusable compiled executor — previous main/QKV rollback

`v16_CompiledBatchExecutor.py` kế thừa nguyên V15. Vòng lặp batch-chunk của
shape #14 tiếp tục nằm ngoài Dynamo để giữ memory bound, nhưng callable xử lý
một sample `[1,S,D]` được `torch.compile(dynamic=False)` đúng một lần rồi dùng
lại cho 32 slice. Compile/autotune xảy ra trong warmup, ngoài CUDA Event timing;
không unroll loop và không giữ đồng thời QKV của nhiều sample.

Compiled callable là inference cache, không thuộc `state_dict`. V16 invalidate
cache sau `load_state_dict()`, `_apply()`/`.to()` và khi đổi train/eval mode.
Training, non-FP32, `S < 8192`, direct call B=1 thông thường và toàn bộ shape
#1–#13 giữ inherited V15/V14.1 behavior. Accuracy harness gọi sample executor
trực tiếp để strict-check đúng graph đã compile thay vì vô tình test eager B=1.

Trên RTX 5090/PyTorch `2.11.0+cu128`, full #14 PASS strict
`0/3,276,800,000`, max abs `0.000944197`. Sandwich measurement
V14.1→V16→V14.1 cho median `7396.7202 → 7166.8359 → 7435.5688 ms`, nên V16
nhanh hơn hai controls `3.11–3.61%`; timed peak giảm `26.977 → 24.487 GiB`.
D-032 từng promote V16; D-038 sau đó promote V16.1 để bỏ exact-#13 branch.
V16 hiện là rollback nếu cần khôi phục direct-layout QKV #13; V15 và V14.1 lần
lượt là QKV-only và eager-memory rollback.

### 10.18 V17 compiled executor batch chunk 2

`v17_CompiledBatch2.py` kế thừa V16 và chỉ tăng
`_LARGE_SEQUENCE_BATCH_CHUNK` từ 1 lên 2. Outer loop vẫn
`torch.compiler.disable`; compiled callable nhận B=2 nên 32 sample official
#14 được xử lý qua 16 calls. Executor gọi trực tiếp V11 forward body để bypass
V14.1 scheduler và tránh recursion, nhưng vẫn giữ V15/V11 arithmetic, cache
lifecycle, cutoff và fallback semantics. B=1 vẫn được support cho batch lẻ.

`shape14_accuracy.py` đọc executor chunk size, chạy optimized group B=2 một
lần rồi compare từng slice với query-blocked reference B=1. Nhờ vậy full gate
thật sự kiểm tra graph B=2. RTX 5090/PyTorch 2.11 full #14 PASS
`0/3,276,800,000`, max abs `0.000944197`. Alternating V16/V17 medians cho V17
nhanh hơn `0.30–0.59%`, trung bình khoảng `0.515%`, timed peak cùng
`24.487 GiB`. Theo D-033, effect quá nhỏ để promote so với V16. D-038 sau đó
đổi main sang V16.1 vì source topology, không phải vì V17.

### 10.19 Shape-#14 inner profiler và external attention probes

`shape14_profile.py` profile trực tiếp reusable inner executor thay vì cấp phát
full B=32. Nó lưu CUDA Event latency, Kineto raw device events, runtime launch
events, peak memory, source hash và heuristic category attribution vào JSON.
Kết quả là diagnostic B=1/B=2, không phải official full-forward latency hay
paired speedup. Backend switch chỉ bao quanh SDPA hiện có để shootout
Flash/cuDNN/Efficient/Math mà không đổi model.

`shape14_fa4_probe.py` và `shape14_sage_probe.py` cô lập exact causal attention
`B1/H16/S100000/Dh64`, dùng QKV distribution thật của V16 khi cần, alternating
CUDA Event timing và strict elementwise comparator. Hai probe không được dùng
làm model speedup; chúng chỉ được phép đề cử một version full-model sau khi vừa
thắng kernel control vừa có error profile đủ hứa hẹn.

Profiler xác nhận attention chiếm `92.258%` V16 inner device time. Built-in
Flash thắng cuDNN `2.38–2.92%`, Efficient gần `2x` chậm; FA4 b28 chậm hơn
PyTorch Flash `7.72%`. Sage PV-FP16/FP32-accum nhanh hơn isolated Flash
`1.3933x` nhưng fail strict attention `94/102.4M`, nên phải qua V18 accuracy
gate và không được dispatch trực tiếp.

### 10.20 V18 SageAttention exact-#14 negative ablation

`v18_SageAttentionShape14.py` kế thừa V16 và chỉ thử thay attention core trên
exact official #14 bằng Sage INT8-QK per-thread, PV-FP16, FP32 accumulation.
Dependency được import optional; CPU, non-target config, training/non-FP32 hoặc
host không có Sage giữ V16. Wrapper dùng `torch.library.custom_op` với fake/meta
output contiguous để Dynamo biết output contract.

Accuracy gate dừng candidate: eager full-model B=1 fail strict đúng
`1/102,400,000`, max abs `0.0026415`. Compiled wrapper còn không equivalent với
eager (`59,375,874/102,400,000` fail), nên không có performance benchmark và
không có dispatch vào `main.py`. V18 được giữ để audit negative result; muốn mở
lại cần recipe precision tốt hơn và `opcheck`/compiled-eager equivalence trước.

### 10.21 V15.1 direct-layout QKV cross-shape ablation

`v15_1_DirectQKVAll.py` kế thừa V15 và chỉ mở rộng construction-time flag:
mọi causal config có `S<8192` đi qua cùng direct-layout QKV operator, còn
large-sequence #14 giữ nguyên V14.1/V16 path. Nó không đổi parameter,
`state_dict`, precision boundary hay fallback training/non-FP32/non-causal.

Official #1–#12 đều PASS strict năm trial. Paired max-autotune theo cả hai
implementation orders cho thấy chỉ #6 (`B=10000,S=128,D=FFN=128`) giảm đồng
thời whole-forward và raw device time: geometric two-order delta lần lượt
`-3.43%` và `-2.99%`. Các shape nhỏ bị host/CUDA-graph floor che khuất; retest
`1000×7` của #2/#4 xác nhận direct path chậm raw GPU `5.52–17.62%`.

V15.1 vì thế là measurement harness, không phải scheduler. Main V16.1 không
chứa direct-QKV; nếu dùng kết quả #6 thì candidate mới phải được xây trên
V16.1, diễn đạt workload regime lớn `B*S` thay vì thêm exact tuple và qua
robustness/aggregate promotion gate riêng.

### 10.22 V16.1 source-clean main

`v16_1_NoDirectQKV13.py` kế thừa trực tiếp V14.1 và chứa riêng reusable compiled
B=1 executor, invalidation/cache lifecycle cùng outer loop đã validate ở V16.
Nó không import/inherit V15, không có direct-QKV flag và không evaluate exact
official-shape tuple. MRO là V16.1→V14.1→V11→V8→V4.3→mixed→baseline.

Official #13 strict năm trial PASS. Paired V14.1/V16.1 sinh cùng 29 kernels,
21 Triton events và operator sequence; host delta đổi dấu `+0.93%/-0.98%` theo
hai orders. Exact-config #14 compiled-executor B=1 canary cũng PASS
`0/102.4M`. D-038 promote V16.1 làm stable main để ưu tiên source topology
không hard-code official tuple. Quyết định này giữ schedule #14 nhưng chủ đích
bỏ direct-QKV #13, nên chấp nhận trả lại win `0.98–2.20%` đã đo của V15/V16.

### 10.23 V17-Sage corrected cross-shape ablation

`v17_sage.py` kế thừa trực tiếp V16.1 và không thay main. Candidate dùng
SageAttention INT8-QK per-thread, PV FP16 với FP32 accumulation cho causal
FP32-eval config có `S>32` và `head_dim<=128`. API pin tự pad head dimension
nhỏ lên 64/128; `head_dim>128`, `S<=32`, non-causal, training, non-FP32, CPU
hoặc thiếu dependency fallback nguyên V16.1. Predicate mô tả support/workload,
không so exact official tuple. Historical `v17_CompiledBatch2.py` vẫn là V17-B2
riêng; alias mới là `v17.sage`/`v17_sage`.

Accuracy recipe dựa trên artifact exact #14 seed 1234: 109 Sage attention
violations đều nằm ở query `1..31`, không còn lỗi ngoài prefix 32. Candidate
vì vậy chạy exact PyTorch Flash trên square causal prefix 32 rồi overwrite 32
rows đầu của Sage output; phần suffix giữ Sage. Attention out-projection dùng
FP16 operands nhưng store FP32 accumulator theo helper V12.1. Custom op mang
tag `cudagraph_unsafe` và large-sequence default dùng
`max-autotune-no-cudagraphs`, vì V18/upstream evidence cho thấy Sage 2.2 cho
output sai khi bị CUDA Graph capture.

Full RTX 5090/PyTorch 2.11 #1–#13 matrix sau đó fail strict ở official #6
(`max_abs=0.00250164`) và #9 (`0.00255397`). #13 PASS và đạt `17.067x` so
baseline, nhưng correctness là gate toàn cục; V17-Sage vì vậy bị reject và giữ
làm negative ablation. V16.1 vẫn là main.

### 10.24 V18-Sage direct automatic performance probe

`v18_sage.py` kế thừa trực tiếp V16.1, không đi qua V17-Sage hoặc historical
V18. Candidate gọi `sageattention.sageattn` automatic. Trên SM120/source pin,
dispatcher chọn INT8-QK per-warp + FP8-PV `fp32+fp16`. Không có exact-prefix,
không copy correction và attention out-projection trở lại đúng arithmetic V16.1.

Sage bật cho causal FP32 CUDA eval với original `head_dim<=128`, gồm official
#12 `S=32`; #8 `head_dim=256` và các branch ngoài support fallback V16.1.
Custom op vẫn mang `cudagraph_unsafe`; shape-#14 executor mặc định dùng
`max-autotune-no-cudagraphs`. Alias mới là `v18.sage`/`v18_sage`, còn `v18`
giữ historical `v18_SageAttentionShape14.py`.

Đây là performance-only diagnostic theo yêu cầu owner. Matrix runner có opt-in
`--benchmark-on-failure` để giữ timing khi strict gate fail, nhưng status vẫn là
`ACCURACY_FAIL`; timing đó không hợp lệ để promote hoặc tính official score.

### 10.25 V16.1 standalone active artifact

`v16_1_clean.py` flatten chuỗi V16.1→V14.1→V11→V8→V4.3→mixed→baseline vào
một module duy nhất. File tự định nghĩa config, parameter topology, reference
fallback, FP16 caches, Flash-first attention, Triton FP32-pre-GELU và reusable
compiled B=1 executor; nó chỉ import PyTorch và optional Triton.

`main.py` là adapter benchmark riêng, còn matrix/profile và shape-#14 tools chỉ
resolve active aliases về main/clean. Toàn bộ 35 version/opcheck files khác nằm
trong `archive/versions/` và không tham gia import graph active. State dict và
local branch/executor outputs đã khớp composed V16.1 bit-for-bit; GPU rerun vẫn
là validation debt, nên packaging này không tạo performance claim mới.

## 11. Điểm mở rộng

- Validate V4.3 Flash-first trên hardware/dtype khác và giữ backend fallback.
- Custom Triton/CUDA cho LayerNorm, QKV projection, attention hoặc FFN fusion.
- FP8 chỉ được xem lại với calibration/QAT hoặc scale/kernel recipe mới đã pass
  strict accuracy; naive per-tensor/MXFP8 inference hiện đã bị loại.
- INT8 FFN-in symmetric per-channel/per-token đã fail ngay official #2, kể cả
  W8-only một layer; chỉ xem lại với outlier/correction recipe khác về bản chất.
- Full FP16 accumulation không được bật global trong submission path: đã fail
  strict accuracy và không tạo speedup sau max-autotune.
- Approximate GELU đã pass accuracy nhưng không được promote khi chưa có
  end-to-end gain vượt measurement noise.
- Autotuning và cache cấu hình theo GPU architecture.
- Tổng hợp/visualize JSON/CSV do matrix runner sinh ra.
- Nsight integration cho memory traffic, kernel launch timeline và occupancy chi tiết.
- Visualizer để so sánh latency/speedup theo shape sau khi có dữ liệu chuẩn hóa.

## 12. Môi trường GPU mục tiêu

Inventory ngày 2026-08-26:

| Thành phần | Giá trị |
|---|---|
| Host OS | Debian GNU/Linux, kernel `6.12.74+deb13+1-amd64` |
| GPU vật lý được chỉ định | GPU index `1` |
| GPU model | NVIDIA GeForce RTX 5090 (GB202) |
| PCI bus | `0000:41:00.0` |
| NVIDIA driver | `595.58.03` |
| Python hệ thống | `3.13.5` |
| PyTorch hệ thống | Không dùng cho project |
| Track 3 venv | `/home/chim/techjam-2026-track3/.venv`, PyTorch `2.13.0+cu130`, CUDA `13.0` |

Chạy workload bằng cách cô lập GPU vật lý số 1:

```bash
source .venv/bin/activate
CUDA_VISIBLE_DEVICES=1 python v3_1_CausalMask.py --device cuda:0 --dtype float32 \
  --batch-size 64 --seq-len 128 --d-model 128 \
  --heads 4 --ffn-dim 128 --layers 4 --causal
```

Sau khi mask bằng `CUDA_VISIBLE_DEVICES=1`, `cuda:0` bên trong process chính là GPU vật lý index `1`.

Tài khoản `chim` hiện thuộc group `gpu`. PyTorch trong venv Track 3 nhận đúng một RTX 5090 khi chạy với `CUDA_VISIBLE_DEVICES=1`; device logic bên trong process là `cuda:0`.
