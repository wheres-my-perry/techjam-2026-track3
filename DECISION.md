# Nhật ký quyết định kỹ thuật

Tài liệu này lưu các quyết định ảnh hưởng dài hạn. Mỗi mục gồm trạng thái, bối cảnh, quyết định, lý do và hệ quả. Thay đổi quyết định bằng một mục mới thay vì xóa lịch sử.

## D-001 — Dùng PyTorch làm reference và harness ban đầu

**Trạng thái:** Accepted  
**Ngày:** 2026-08-26

### Bối cảnh

Đề cho phép chọn PyTorch hoặc TensorFlow. Repository hiện có `torch_transformer_benchmark.py` hoàn chỉnh và `v1_fuseQKV.py` dựa trên PyTorch.

### Quyết định

Dùng nhánh PyTorch làm đường triển khai chính. Không duy trì hai framework trước khi nhánh PyTorch được validate và benchmark đầy đủ.

### Lý do

- Giảm gấp đôi test/optimization surface.
- PyTorch cung cấp SDPA, `torch.compile`, CUDA Event và đường mở rộng Triton/CUDA phù hợp.
- Code hiện tại đã dùng PyTorch end-to-end.

### Hệ quả

Submission chỉ cần benchmark Torch nếu luật cuối cùng vẫn cho phép chọn một trong hai. Nếu benchmark chính thức thay đổi, quyết định này phải được xem lại.

## D-002 — Baseline là oracle, không phải optimization target

**Trạng thái:** Accepted  
**Ngày:** 2026-08-26

### Bối cảnh

Speedup chỉ có ý nghĩa nếu baseline và optimized thực hiện cùng phép tính, cùng trọng số và cùng input.

### Quyết định

Giữ `torch_transformer_benchmark.py` làm reference. Optimization nằm trong implementation phiên bản hóa. Không đổi baseline, workload hoặc luật đo để cải thiện score.

### Hệ quả

- Dễ audit correctness và benchmark fairness.
- Một thay đổi bắt buộc từ benchmark chính thức phải được import rõ ràng và ghi vào tài liệu, không trộn với optimization.

## D-003 — Correctness gate trước performance

**Trạng thái:** Accepted  
**Ngày:** 2026-08-26

### Bối cảnh

GPU kernels có thể nhanh nhưng sai do mask semantics, reduction order hoặc low-precision accumulation.

### Quyết định

Accuracy phải pass trước khi kết quả latency/speedup được xem là hợp lệ. Kiểm tra bao phủ nhiều seed, shape, dtype, causal và padding. `--benchmark-on-failure` chỉ phục vụ debug.

### Hệ quả

- Một candidate nhanh nhưng fail một case mục tiêu không được đưa vào scheduler.
- Kết quả performance phải gắn với accuracy result của cùng implementation/configuration.

## D-004 — `v1` chỉ fuse QKV projection

**Trạng thái:** Accepted for experimentation
**Ngày:** 2026-08-26

### Bối cảnh

Ba Q/K/V projection riêng tạo ba GEMM và đọc cùng input ba lần.

### Quyết định

`v1_fuseQKV.py`:

- Pack QKV weights/biases thành một projection.
- Chạy một `F.linear` rồi tách output thành Q, K, V.
- Giữ nguyên attention math của baseline.
- Khi training, dùng lại ba projection gốc.
- Refresh packed cache sau `load_state_dict()`.

### Lý do

Đây là thay đổi nhỏ, dễ đọc và cô lập đúng chi phí của QKV projection.

### Hệ quả

- Packed cache thêm lifecycle cần kiểm tra.
- Chưa tối ưu attention, mask, softmax, output projection hoặc FFN.
- Quyết định chỉ được promote sau full GPU accuracy/performance matrix.

## D-005 — Phiên bản hóa các phương án thử nghiệm

**Trạng thái:** Accepted  
**Ngày:** 2026-08-26

### Bối cảnh

Tối ưu GPU phụ thuộc shape/hardware và thường có regression. Ghi đè implementation cũ làm mất baseline thực nghiệm.

### Quyết định

Giữ candidate ở các file `vN.py` hoặc module có version rõ ràng cho đến khi có dữ liệu để promote. Mỗi version phải có entry trong `EXPERIMENTS.md`.

### Hệ quả

- Có thể tái chạy và so sánh lịch sử.
- Có một lượng duplication tạm thời; chỉ refactor phần chung sau khi API ổn định.

## D-006 — Kết quả benchmark phải tái lập được

**Trạng thái:** Accepted  
**Ngày:** 2026-08-26

### Bối cảnh

Latency GPU thay đổi theo GPU model, driver, CUDA/PyTorch, clock/thermal state, dtype, shape, warmup, compile và TF32.

### Quyết định

Không ghi một speedup đơn lẻ. Mỗi kết quả phải kèm:

- Git revision hoặc hash của implementation.
- GPU, compute capability, driver, CUDA, cuDNN, PyTorch và OS.
- Full command/configuration.
- Baseline/optimized median, p90, throughput và speedup.
- Warmup, repeats, rounds và seed.
- Accuracy thresholds và kết quả.

### Hệ quả

Technical report có audit trail. Các số không có metadata được xem là quan sát tạm, không phải kết quả chính thức.

## D-007 — Scheduler chỉ chứa candidate đã validate

**Trạng thái:** Proposed  
**Ngày:** 2026-08-26

### Bối cảnh

Shape-aware dispatch có thể cải thiện aggregate score nhưng tăng nhánh correctness và overhead.

### Quyết định đề xuất

Scheduler dùng key gồm hardware, dtype, shape và mask semantics. Chỉ candidate đã pass full accuracy coverage mới được đăng ký. Case không biết dùng safe fallback.

### Hệ quả dự kiến

- Cần benchmark matrix runner và registry metadata.
- Autotuning nên chạy offline; dispatch online phải rẻ và deterministic.
- Chưa triển khai cho đến khi danh sách shape chính thức và kết quả `v1` có sẵn.

## D-008 — Cố định benchmark trên GPU vật lý index 1

**Trạng thái:** Accepted  
**Ngày:** 2026-08-26

### Bối cảnh

Máy benchmark có hai NVIDIA GeForce RTX 5090. Dự án được chỉ định sử dụng GPU vật lý index `1`, PCI bus `0000:41:00.0`.

### Quyết định

Mọi accuracy/performance run chính thức của task dùng:

```bash
CUDA_VISIBLE_DEVICES=1 python3 v1_fuseQKV.py --device cuda:0 ...
```

Trong process đã mask, `cuda:0` ánh xạ tới GPU vật lý index `1`.

### Hệ quả

- Không chạy benchmark trên GPU vật lý index `0` hoặc chia workload qua cả hai GPU.
- Metadata kết quả phải ghi model, GPU index và PCI bus để phát hiện nhầm thiết bị.
- Tài khoản hiện chưa thuộc group `gpu`; phải giải quyết quyền device trước khi chạy.
- Máy có một environment PyTorch CUDA của project khác; Track 3 vẫn cần environment riêng, dùng phiên bản đã tương thích làm mốc.

## D-009 — Import nguyên bản benchmark Torch cập nhật ngày 2026-08-27

**Trạng thái:** Accepted

**Ngày:** 2026-08-27

### Bối cảnh

Ban tổ chức đánh dấu `torch_transformer_benchmark.py` là đã cập nhật cùng Appendix test shapes. Attachment mới dùng phép so sánh `<=`; trang đề bài vẫn ghi strict `<`, và docstring/default CLI trong attachment không đồng nhất về giá trị tolerance.

### Quyết định

Lưu baseline đúng byte của attachment, SHA-256 `5529c96a80799b51f68092e1444a30b17994554dffdf52da98ba701489a7f36e`. Không tự sửa baseline để hòa giải mâu thuẫn; `v1_fuseQKV.py` theo comparator `<=` và giữ CLI default thực tế `rtol=0.02`, `atol=0.002`.

### Hệ quả

- Có thể chứng minh baseline khớp file chính thức đã tải.
- Technical report phải nêu rõ khác biệt giữa trang đề bài và attachment.
- Nếu ban tổ chức phát hành clarification mới, cập nhật bằng một decision mới thay vì sửa lịch sử.

## D-010 — Chốt correctness theo trang đề bài

**Trạng thái:** Accepted; supersedes comparator choice in D-009  
**Ngày:** 2026-08-27

### Bối cảnh

Trang đề bài ghi strict `<` với `relative error < 0.02 OR absolute error < 0.002`, trong khi attachment Torch ngày 2026-08-27 dùng `<=` và có docstring tolerance không khớp CLI.

### Quyết định

Chuẩn hóa cả `torch_transformer_benchmark.py` và `v1_fuseQKV.py` về strict `<`, `rtol=0.02`, `atol=0.002`. Giữ source hash của attachment trong tài liệu để audit nguồn.

### Hệ quả

- Accuracy gate khớp đúng câu chữ trên trang đề và yêu cầu đã chốt trong repository.
- Baseline local không còn byte-identical với attachment; khác biệt chỉ nằm ở comparator, thông báo criterion và docstring tolerance.

## D-011 — Mỗi version cô lập một bước tối ưu

**Trạng thái:** Accepted

**Ngày:** 2026-08-27

### Bối cảnh

Bản `v2_SPDA.py` trước đây đã cộng dồn SDPA, packed-QKV no-copy và flattened whole-model loop. Cách đặt version này làm mất một ablation dễ đọc để phân biệt speedup do SDPA với speedup do data movement và Python/module dispatch.

### Quyết định

- `v1_fuseQKV.py`: chỉ fuse ba Q/K/V projection.
- `v2_SPDA.py`: lấy v1 làm nền và chỉ thay explicit attention math bằng SDPA trên FP32 path.
- `v3_SDPA_NoCopy.py`: thêm Q/K/V views không-copy, mask reuse và flattened whole-model loop.

### Hệ quả

- Mỗi version có một giả thuyết hiệu năng rõ ràng và có thể benchmark độc lập.
- V1/v2 được giữ làm ablation; v3 là candidate nhanh nhất tại thời điểm D-011 (superseded by D-014).
- Khi thuật toán thay đổi hoặc có version mới, phải cập nhật đồng thời `EXPERIMENTS.md`, `SOLUTION.md`, kiến trúc và kế hoạch triển khai.

## D-012 — Performance benchmark chỉ dùng test shapes chính thức

**Trạng thái:** Accepted

**Ngày:** 2026-08-27

### Bối cảnh

Repository từng dùng default shape `(B=8, S=128, D=512, H=8, FFN=2048, L=6, non-causal)` để phát triển và ablation. Shape này không nằm trong 14 test shapes được công bố ở Appendix của đề nên không đại diện trực tiếp cho workload chấm điểm.

### Quyết định

Từ quyết định này, performance benchmark chỉ hợp lệ khi `B`, `S`, `D`, `H`, số layer, FFN dimension và causal flag khớp chính xác một test shape chính thức. Bảng shape không chỉ định padding nên performance dùng no-padding mặc định cho đến khi có clarification mới.

Shape khác vẫn có thể dùng cho correctness, debug, profiler hoặc ablation, nhưng phải gắn nhãn **non-official diagnostic** và không được dùng làm headline result.

### Hệ quả

- Kết quả chính hiện tại chỉ gồm official shape #1; các default non-causal result được giữ như lịch sử diagnostic.
- Mọi benchmark mới phải ghi official shape ID trong command/result.
- Full matrix runner phải lấy trực tiếp 14 rows trong `STATEMENT.md`, không tự sinh performance shapes khác.

## D-013 — Matrix runner cách ly từng shape bằng subprocess

**Trạng thái:** Accepted

**Ngày:** 2026-08-27

### Bối cảnh

Các official shapes có chênh lệch workload rất lớn; shape #6 và #14 có thể OOM, bị OS kill hoặc vượt thời gian chạy. Chạy toàn bộ matrix trong một Python process sẽ làm mất kết quả trước đó khi một case phá vỡ process.

### Quyết định

`matrix_runner.py` chạy từng official shape bằng một subprocess độc lập, áp timeout theo case và ghi JSON/CSV sau mỗi lần chạy. Runner tiếp tục qua accuracy failure, nonzero exit, OOM/kill và timeout; exit code cuối khác 0 nếu matrix không PASS toàn bộ.

### Hệ quả

- Partial results vẫn tồn tại khi shape lớn thất bại.
- Mỗi row giữ full command và raw output để audit.
- Process startup không nằm trong CUDA Event latency của child benchmark.
- `benchmark-results/` là artifact sinh tự động và không được commit mặc định.

## D-014 — V3.1 dùng causal flag thay cho materialized causal mask

**Trạng thái:** Accepted as current FP32 candidate; full matrix pending
**Ngày:** 2026-08-28

### Bối cảnh

V3 kết hợp key-padding mask và lower-triangular mask thành `[B,1,S,S]`. Allocation này tăng theo `B×S²`, làm giảm hiệu năng ở sequence dài và không khả thi cho shape #14. V3 còn zero invalid attention output trước khi toàn block tiếp tục zero cùng query position.

### Quyết định

Tạo `v3_1_CausalMask.py` giữ nguyên v3 nhưng truyền key mask trực tiếp cùng `is_causal=config.causal`; bỏ causal-mask buffer, phép kết hợp mask và lần `masked_fill` trong attention. Vẫn zero invalid query cuối mỗi block và sau final LayerNorm.

### Hệ quả

- Causal masking không còn tạo intermediate theo `B×S²` trong user implementation.
- Padding semantics phải được kiểm tra riêng trên causal/non-causal và nhiều seed trước benchmark.
- V3 được giữ nguyên làm ablation; v3.1 đã qua core accuracy gate và official benchmark #1/#13, nhưng vẫn cần full matrix trước submission.

## D-015 — Tách attribution eager ATen và compiled device events

**Trạng thái:** Accepted
**Ngày:** 2026-08-28

### Bối cảnh

Profiler ban đầu gắn `record_function` quanh `F.linear`, LayerNorm, GELU và SDPA để quy self CUDA time về model stage. Cách này phù hợp eager nhưng không còn đáng tin sau `torch.compile`: Inductor có thể fuse nhiều ATen operator thành một Triton kernel, còn monkey-patch profiler-only sau compile có thể làm graph recompile hoặc thay đổi chính graph cần đo. Official shape #1 cho thấy v3.1 `reduce-overhead` giảm median từ `0.5118 ms` eager xuống `0.3169 ms`, nên compiled-path attribution trở thành yêu cầu của technical report.

### Quyết định

`profile_models.py` giữ hai schema riêng:

- Eager dùng ATen category và non-overlapping model-stage scopes hiện tại.
- Compiled dùng raw GPU device events và runtime evidence: device/kernel/memory-event count, Triton event/launch count, `Torch-Compiled Region`, CUDA Graph launch API, top device events và steady/peak CUDA allocation.
- Child compiled bật unique Inductor kernel names và lưu backend, mode cùng mode options thực tế vào JSON.
- Không quy một fused compiled kernel ngược về LayerNorm/GELU/elementwise nếu trace không cung cấp bằng chứng trực tiếp.

### Hệ quả

- Eager và compiled profile có thể so kernel count, device time và launch gaps nhưng không cộng/trộn ATen stage share với fused-kernel share.
- Attribution graph fusion so với CUDA Graph cần ablation `default` versus `reduce-overhead`; headline speedup đơn lẻ chưa đủ kết luận cơ chế.
- Chrome trace/Nsight vẫn là nguồn chi tiết khi kernel name hoặc CUDA Graph replay không đủ rõ trong bảng terminal.

## D-016 — V4.2 dispatch SDPA theo shape với automatic fallback

**Trạng thái:** Accepted for RTX 5090 shapes #1–#13; shape #14 pending
**Ngày:** 2026-08-28

### Bối cảnh

V4.1 compiled mặc định dùng memory-efficient SDPA. Forced-backend ablation trên
RTX 5090 cho thấy cuDNN nhanh hơn rõ ở một số shape, đặc biệt giảm official shape
#13 từ `2.3126` xuống `1.6841 ms`, nhưng chậm hơn ở #6/#11/#12 và không hỗ trợ
shape #8 có head dimension `256`. Flash Attention không nhận non-null
`attn_mask`, còn forced Efficient khớp automatic dispatch.

### Quyết định

`v4_2_SDPA_Dispatch.py` chỉ ép `CUDNN_ATTENTION` cho official shapes
#1/#2/#3/#4/#7/#9/#13 đã có measured win. Các shape khác hoặc chưa biết giữ
automatic PyTorch SDPA. Dispatch dựa trên Python config trước compile, không đọc
tensor hoặc thêm data-dependent branch vào hot path.

### Hệ quả

- Candidate PASS strict accuracy trên #1–#13; causal/non-causal padding branch
  của cuDNN PASS 5 trial trên diagnostic shape #1.
- Geomean speedup #1–#13 tăng từ khoảng `7.09x` ở V4.1 lên `7.58x` ở V4.2.
- Bảng dispatch chỉ được coi là hợp lệ cho RTX 5090/PyTorch `2.13.0+cu130` đã
  benchmark; hardware hoặc software khác phải dùng V4.1 automatic dispatch hoặc
  được benchmark lại trước khi bật V4.2 mapping.
- Shape #14 vẫn pending và không được bỏ khỏi final correctness requirement.

## D-017 — V4.3 bỏ key mask dư trên causal right-padding Flash path

**Trạng thái:** Accepted for RTX 5090 shapes #1–#13; shape #14 pending
**Ngày:** 2026-08-28

### Bối cảnh

V4.2 forced Flash thất bại vì harness truyền non-null `attn_mask` kể cả khi mask
toàn `True`. Với causal self-attention và right padding, valid tokens tạo một
prefix: valid query chỉ attend các key trước hoặc tại vị trí của nó, nên không
thể nhìn padded keys nằm trong suffix. Invalid query outputs đã được zero cuối
mỗi block. Vì vậy key-padding attention mask là dư trên nhánh này.

Forced Flash với `attn_mask=None` PASS official #1–#13 và giảm attention kernel
#13 từ `0.9607` xuống `0.4232 ms` so với V4.2 cuDNN. Flash không thắng mọi shape;
#6 chậm hơn automatic và #2/#3/#9 chỉ có chênh lệch nhỏ.

### Quyết định

Tạo `v4_3_SDPA_CausalFlash_Dispatch.py`:

- Flash/no key mask cho #1/#4/#5/#7/#8/#10/#11/#13.
- Masked cuDNN cho #2/#3/#9.
- Masked automatic PyTorch SDPA cho #6/#12 và config lạ.
- Chỉ áp causal mask-elision khi config nằm trong bảng đã validate; non-causal
  và unknown shapes giữ mask.

### Hệ quả

- Direct matrix #1–#13 PASS strict accuracy; geomean tăng `7.58x → 8.48x` so
  với V4.2. Shape #13 đạt `1.1412 ms`, `36.604x`.
- Causal right-padding diagnostics 25% và 75% đều PASS, nhưng phép chứng minh
  không áp dụng cho arbitrary masks có lỗ hoặc valid tokens không tạo prefix.
- Bảng dispatch chỉ hợp lệ cho RTX 5090/PyTorch `2.13.0+cu130` đã đo.
- Shape #14 vẫn pending và không bị bỏ khỏi final requirement.

## D-018 — Đơn giản hóa V4.3 thành Flash-first với backend fallback

**Trạng thái:** Accepted; supersedes the static V4.3 mapping in D-017
**Ngày:** 2026-08-28

### Bối cảnh

Forced Flash PASS #1–#13. Static dispatcher D-017 tối ưu #6 bằng automatic và
một số shape nhỏ bằng cuDNN, nhưng geomean đo được chỉ `8.48x`. Direct all-Flash
run của file đơn giản đạt `8.52x`; regression #6 (`26.29 → 26.96 ms`, khoảng
2.5%) được bù bởi các shape khác. Bảng shape dài không còn tạo lợi ích aggregate
và làm implementation khó trình bày hơn.

PyTorch `2.13.0+cu130` trên target cung cấp
`sdpa_kernel(backends, set_priority=True)`, cho phép khai báo thứ tự backend và
tự bỏ qua backend không đủ điều kiện.

### Quyết định

Promote `v4_3_Flash.py` làm V4.3 candidate:

- Mọi causal/right-padding optimized path bỏ redundant key attention mask.
- Backend priority: Flash → cuDNN → Efficient → Math.
- Non-causal giữ key mask và automatic PyTorch dispatch.
- Training hoặc public input khác FP32 giữ full-reference fallback.
- Các V4.3 static/forced files được giữ như historical ablation, không phải file
  submission chính.

### Hệ quả

- Direct matrix #1–#13 PASS strict accuracy; geomean `8.52x`.
- Shape #13 đạt `1.1412 ms`, `36.604x`; profiler xác nhận
  `pytorch_flash::flash_fwd_kernel` và một CUDA Graph launch mỗi forward.
- CPU causal diagnostic chứng minh Math fallback PASS; non-causal + padding
  diagnostic PASS masked automatic path.
- Shape #6 chậm hơn static V4.3 khoảng 2.5%; đây là trade-off được chấp nhận để
  có implementation ngắn, portable hơn và không hard-code official shapes.
- Shape #14 và arbitrary sparse-mask semantics vẫn pending/ngoài fast-path proof.

## D-019 — Không promote naive FP8 inference khi chưa có calibration/QAT

**Trạng thái:** Accepted
**Ngày:** 2026-08-28

### Bối cảnh

RTX 5090 hỗ trợ E4M3 FP8 GEMM và native MXFP8 block scaling. Target PyTorch
`2.13.0+cu130` chạy đúng per-tensor GEMM và MXFP8 all-ones sanity test, nhưng
SDPA không nhận FP8. Row-wise `_scaled_mm` còn trả sai ngay sanity test trên
runtime này. Per-tensor V5, full MXFP8 và mọi projection-scope MXFP8 ablation
đều FAIL strict model accuracy trên official shape #2; biến thể nhẹ nhất là
QKV-only vẫn fail `7,532/49,152` phần tử qua ba trial.

### Quyết định

Không đưa V5/V5.1-MXFP8/V5.2 vào scheduler hoặc bảng performance. Không benchmark
candidate sai bằng `--benchmark-on-failure` để tạo speedup claim. V4.3 FP16 vẫn
là candidate chính. Chỉ mở lại FP8 khi có calibration/QAT, scale recipe/runtime
đã qua kernel sanity, hoặc một dispatch scope cụ thể pass strict multi-seed/full
shape coverage trước performance.

### Hệ quả

- Giữ hồ sơ FP8 như negative ablation và bằng chứng technical report.
- Peak FP8 Tensor Core throughput không được xem là bằng chứng end-to-end nếu
  dynamic quantization overhead và accuracy chưa được đo.
- Không cài thêm `torchao` chỉ để che failure: built-in kernel đã được probe;
  recipe mới vẫn phải chứng minh correctness độc lập trên target.

## D-020 — Không promote full FP16 accumulation; dành alias V5.1 cho ablation mới

**Trạng thái:** Accepted
**Ngày:** 2026-08-28

### Bối cảnh

V4.3 dùng FP16 cho GEMM nhưng mặc định accumulate FP32. Bật
`torch.backends.cuda.matmul.allow_fp16_accumulation=True` có thể tăng throughput
trên RTX 5090, nhưng đây là flag process-global và đánh đổi numerical accuracy.
Repository trước đây từng gọi candidate MXFP8 đã xóa là V5.1, không còn alias
runnable nào mang tên đó.

### Quyết định

- Tạo `v5_1_FP16Accum.py`; alias `v5.1` trỏ tới file này. Hồ sơ MXFP8 cũ được
  gọi rõ là `V5.1-MXFP8`, không sửa hoặc ghi đè kết quả lịch sử.
- Giữ implementation như negative ablation, không promote và không dispatch
  theo shape: max-autotune fail official #10, reduce-overhead fail official #8.
- Không dùng `--benchmark-on-failure`. Chỉ latency từ row/candidate đã PASS mới
  được dùng để chẩn đoán, không tính aggregate score cho matrix lỗi.

### Hệ quả

- Paired max-autotune shape #8 cho V4.3 `2.5258 ms`, V5.1 `2.5336 ms`; cùng
  kernel mix nên không có performance evidence ủng hộ flag sau autotune.
- Reduce-overhead shape #8 fail `40,029/41,943,040` phần tử; max-autotune shape
  #10 fail `1/5,242,880` phần tử.
- V4.3 + max-autotune vẫn là candidate tốt nhất; muốn thử reduced accumulation
  khác phải tạo version mới và qua strict accuracy matrix trước benchmark.

## D-021 — Không promote approximate GELU nếu không tạo end-to-end gain

**Trạng thái:** Accepted
**Ngày:** 2026-08-28

### Bối cảnh

V4.3 max-autotune vẫn có GELU trong FFN. Tanh approximation rẻ hơn exact/erf về
mặt biểu thức và có thể dùng một phần error budget còn lại, nhưng Inductor có
thể đã fuse exact GELU vào GEMM/LayerNorm epilogue nên micro-op rẻ hơn không đảm
bảo whole-model latency thấp hơn.

### Quyết định

Giữ `v6_ApproxGELU.py` làm one-variable ablation và alias `v6`, nhưng không
promote. Candidate PASS strict official #1–#13; performance phải được đánh giá
theo paired end-to-end latency, không dựa trên chi phí lý thuyết của tanh.

### Hệ quả

- Clean paired #2: host median `0.0780 ms` V4.3 và `0.0749 ms` V6, nhưng raw
  device time lại `0.0554 → 0.0560 ms`; V6 còn tăng từ 32 lên 33 kernels vì
  mất một epilogue fusion. Tín hiệu trái nhau nên không phải gain hợp lệ.
- Clean paired #8: V4.3 `2.5502 ms`, V6 `2.5584 ms`; V6 chậm hơn `0.32%`.
- Reverse-order retries bị một evaluation workload mới làm nhiễu nên bị loại
  khỏi claim; các lượt cũ dưới training contention chỉ dao động quanh zero.
- Full matrix timing không dùng để kết luận vì một training process khác chiếm
  `53–80%` GPU vật lý #1; accuracy của cùng matrix vẫn hợp lệ.
- V4.3 + max-autotune tiếp tục là best path. Chỉ reconsider V6 nếu một thay đổi
  khác khôi phục epilogue fusion và tạo gain ổn định lớn hơn variance.

## D-022 — Không thay Inductor residual/LayerNorm fusion bằng custom Triton khi codegen đã tương đương

**Trạng thái:** Accepted
**Ngày:** 2026-08-28

### Bối cảnh

V4.3 source trả projection output về FP32 trước residual và gọi LayerNorm ở
operation kế tiếp. V7a pipeline graph thành residual FP32 cùng normalized FP16
để expose cast + add + optional mask + LayerNorm cho compiler. Nếu Inductor còn
materialize intermediate hoặc launch kernel riêng, một custom Triton dual-output
kernel có thể giảm memory traffic.

### Quyết định

Giữ `v7_ResidualLayerNorm.py` làm pure-PyTorch ablation nhưng không promote và
không triển khai V7b custom Triton. Chỉ mở lại custom residual/LayerNorm nếu một
runtime/shape khác cho profiler evidence về unfused kernel hoặc materialization
đủ lớn.

### Hệ quả

- Local official #2 và bốn nhánh causal/non-causal × padding/no-padding PASS;
  V7a khớp V4.3 eager bit-for-bit. GPU strict #7/#10 cũng PASS 5/5.
- Max-autotune #2/#8/#12 cho V4.3 và V7a cùng kernel/Triton-event count và cùng
  event names. Device time #2 bằng nhau; #8 V7a chậm `0.32%`; #12 effect đổi
  dấu khi đảo thứ tự.
- Compiled event đã chứa `addmm + cast + residual + LayerNorm` và biến thể
  `mask + LayerNorm`. Standalone Triton kernel có nguy cơ de-fuse GEMM template,
  tăng launch hoặc memory traffic thay vì giảm.
- V4.3 + max-autotune tiếp tục là best path; hướng tiếp theo phải nhắm phần còn
  chiếm device time như GEMM/FFN, không lặp lại fusion compiler đã làm.

## D-023 — Chỉ dispatch custom FFN-in/GELU cho large-token D=FFN=128

**Trạng thái:** Accepted
**Ngày:** 2026-08-28

### Bối cảnh

Profiler V4.3 cho thấy exact GELU vẫn là kernel riêng ở một số graph. V8a fuse
FFN-in GEMM, bias và erf-GELU trong Triton, giữ FP32 dot accumulation cùng FP16
rounding boundary của V4.3. Lợi ích phụ thuộc mạnh vào shape: isolated #8 có
microkernel win lớn nhưng whole-model gần như hòa, trong khi #6 có đủ token để
phần launch/intermediate saving vượt khác biệt GEMM template.

### Quyết định

- Giữ `v8_FusedFFNGELU.py` như candidate shape-dispatched trên V4.3.
- Chỉ bật custom op khi `B*S >= 1_000_000`, `D=128`, `FFN=128`; config khác gọi
  nguyên `_mixed_ffn` V4.3.
- Không mở rộng dispatch chỉ dựa trên isolated microbenchmark. Mỗi shape mới phải
  có paired whole-model result, strict accuracy và order/noise check.
- Giữ exact GELU và FP16 rounding point; không đổi accumulation precision hay
  dùng tanh approximation trong cùng ablation.

### Hệ quả

- Paired max-autotune official #6 giảm `26.6799 → 25.4092 ms` (`-4.76%`
  latency), raw GPU time `26.6680 → 25.3948 ms`; kernel count `32 → 29`.
- Hai paired order trước đó cũng cho `-4.75%` và `-4.94%`, nên effect ổn định.
- Official #8 giảm ba kernels nhưng end-to-end chỉ dao động quanh zero; fallback
  tránh regression/noise ở case này và mọi shape nhỏ.
- Full official #1–#13 V8 matrix PASS 5/5 mỗi shape, max abs lớn nhất
  `0.00188218`; shape #14 vẫn pending.
- V8 là best path cho large-token D=FFN=128; V4.3 vẫn là portability/safe
  fallback cho phần còn lại.

## D-024 — Không thay V8 dispatcher bằng force-all dù aggregate latency giảm nhẹ

**Trạng thái:** Accepted
**Ngày:** 2026-08-29

### Bối cảnh

V8 ban đầu chỉ bật custom FFN/GELU ở official #6. V8.1 ép cùng path trên
#1–#13 để kiểm tra xem kernel-count reduction có chuyển thành aggregate win và
có đủ ổn định để bỏ dispatcher hay không.

### Quyết định

- Giữ `v8_1_FusedFFNGELUAll.py` làm force-all ablation, không ghi đè V8 stable.
- Không dùng unconditional fusion làm submission path lúc này.
- Có thể mở rộng V8 dispatcher cho từng shape chỉ khi latency giữ dấu ở cả hai
  order, raw GPU time đồng thuận và gain đủ lớn hơn measurement quantization.

### Hệ quả

- Strict #1–#13 PASS 5/5 mỗi shape, max abs `0.00188218`.
- Geomean paired latency giảm `1.365%` theo order V4.3→V8.1 và `0.969%` theo
  reverse order; force-all có aggregate benefit nhỏ.
- #1/#3/#4/#5/#10 giữ dấu win khoảng `1.2–2.7%`; #6 giữ win khoảng `4.8%`.
- #2 reverse-order host latency regress `2.93%`; #12 đổi từ `-2.52%` sang
  `+2.47%`; #11 raw GPU regress `0.41–0.52%`. Do đó force-all không Pareto-win
  và aggregate geomean không đủ biện minh regression per-shape.
- Dispatcher tiếp theo nên whitelist measured winners và giữ V4.3 cho
  #2/#11/#12; các effect dưới `1.5%` cần measurement dài hơn.

## D-025 — Không promote fully fused persistent MLP chỉ từ isolated win

**Trạng thái:** Accepted
**Ngày:** 2026-08-29

### Bối cảnh

V8 vẫn materialize hidden activation và gọi FFN-out GEMM riêng. V9 giữ hidden
tiles on-chip và fuse cả `FFN-in → exact GELU → FFN-out`, kỳ vọng đặc biệt có
lợi ở shape nhỏ hoặc narrow FFN.

### Quyết định

- Giữ `v9_PersistentMLP.py` làm fully fused ablation cho D/FFN≤128.
- Không thêm V9 vào best-path dispatcher hiện tại.
- Isolated microkernel speedup chỉ là gate để chạy whole model, không phải bằng
  chứng promotion. Primary metric vẫn là paired end-to-end latency giữ dấu qua
  order check.

### Hệ quả

- V9 kernel PASS comparator và nhanh hơn compiled two-GEMM reference
  `1.18–1.59x` trên unique supported official FFN shapes.
- Full #1–#13 strict accuracy PASS 5/5; max abs `0.00188218`.
- Whole-model #1/#5/#13 regress `4.7%/2.0%/5.3%`; #2/#12 hòa.
- #7 raw GPU time giảm khoảng `12%` và kernels `32→25`, nhưng primary latency
  đổi dấu theo implementation order nên không phải stable score win.
- #6 V9 `25.4481 ms` chậm hơn V8 `25.4071 ms` dù chỉ còn 25 thay vì 29 kernels.
- Compiler GEMM templates, existing graph fusion và timing floor quan trọng hơn
  launch count. Chỉ mở lại V9 nếu tile schedule mới tạo paired whole-model win.

## D-026 — V11 chỉ là precision whitelist candidate, không thay V8.1 force-all

**Trạng thái:** Accepted
**Ngày:** 2026-08-29

### Bối cảnh

V8/V8.1 accumulate FFN-in GEMM ở FP32 nhưng round Linear output xuống FP16
trước exact GELU. V11 bỏ round này mà vẫn store GELU FP16 cho FFN-out, nên không
thêm kernel hoặc memory traffic. Candidate force custom path mọi shape để đo
trực tiếp accuracy/latency trade-off với V8.1.

### Quyết định

- Giữ `v11_FP32PreGELU.py` làm force-all precision ablation; không thay V8/V8.1
  unconditional trong submission path.
- Xem #6/#10 là whitelist candidates vì paired accuracy và latency đồng thuận;
  không whitelist #7 nếu mục tiêu là Pareto latency.
- Chỉ tạo/promote dispatcher V11 sau shape #14 và robustness matrix. Không suy
  rộng kết quả sang shape chưa paired chỉ từ full accuracy PASS.

### Hệ quả

- RTX 5090 max-autotune #1–#13 strict PASS, failed `0`; worst max abs giảm từ
  V8.1 `0.00188218` xuống V11 `0.00179082`.
- #6 max abs giảm `13.01%`; host latency chỉ đổi `+0.10%/+0.02%` qua hai order,
  raw GPU `+0.003%/-0.064%`, nên được xem là latency trung tính.
- #10 max abs giảm `1.19%`; host latency giảm `1.50%` ở cả hai order và raw GPU
  giảm `1.11–1.17%`, tạo Pareto improvement đã đo.
- #7 max abs giảm `4.85%` nhưng host latency tăng `1.47–2.77%` và raw GPU tăng
  `0.10–0.74%`; accuracy gain không đủ để gọi unconditional Pareto win.
- Cả hai path giữ 29 GPU kernels, 2 memory events, 21 Triton events và một CUDA
  Graph launch/forward; khác biệt latency đến từ arithmetic/codegen chứ không
  phải launch count.

## D-027 — Promote V11 force-all thành main implementation

**Trạng thái:** Accepted; supersedes promotion choice in D-026
**Ngày:** 2026-08-29

### Bối cảnh

V11 đã PASS strict #1–#13 và giảm worst max absolute error so với V8.1. D-026
ưu tiên Pareto latency nên chỉ xem #6/#10 là whitelist candidates, nhưng project
owner quyết định ưu tiên accuracy margin và implementation thống nhất hơn, chấp
nhận regression nhỏ đã đo ở #7.

### Quyết định

- Promote nguyên force-all `v11_FP32PreGELU.py` thành main optimized implementation.
- `main.py` là stable entrypoint mỏng trỏ tới V11; aliases `main` và `best` cùng
  default của matrix runner trỏ entrypoint này.
- Không đổi thuật toán V11, baseline, tolerance, workload hoặc accuracy gate.
- V8/V8.1 tiếp tục tồn tại làm performance/precision controls và rollback path.

### Hệ quả

- Main ưu tiên exact GELU từ FP32 accumulator trên mọi mixed-precision shape,
  kể cả #7 nơi host latency regress `1.47–2.77%` so với V8.1.
- Đổi lại, V11 giảm max abs `4.85%` ở #7, `13.01%` ở #6 và `1.19%` ở #10;
  #6 latency trung tính còn #10 nhanh hơn khoảng `1.50%`.
- Full main claim vẫn chỉ là #1–#13; shape #14 và robustness matrix còn pending.
- Promotion là lựa chọn sản phẩm/accuracy-risk có chủ đích, không phải tuyên bố
  V11 Pareto-win trên mọi shape.

## D-028 — Không triển khai kernel cho symmetric INT8 FFN-in recipe hiện tại

**Trạng thái:** Accepted
**Ngày:** 2026-08-29

### Bối cảnh

INT8 Tensor Core GEMM có thể nhanh hơn FP16 nếu cả activation và weight đều
INT8, nhưng dynamic activation quantization, dequantization và numerical error
phải được chứng minh end-to-end. V13 vì thế mô phỏng số học trước khi viết
kernel: weight symmetric per-output-channel, activation dynamic symmetric
per-token, INT32 accumulator, dequantize FP32 trước exact GELU; attention,
FFN-out và mọi boundary khác giữ V11.

Trên local PyTorch `2.12.1`, official shape #2 qua năm seed fail cả ba control:
W8-only `1,659/81,920`, A8-only `4,574/81,920`, W8A8 `5,825/81,920`; max abs
lần lượt `0.00738502`, `0.0122206`, `0.0198889`. Chỉ quantize W8 ở layer cuối
vẫn fail `69/81,920`, `max_abs=0.00381088`; W8A8 layer cuối fail
`1,332/81,920`, `max_abs=0.00912209`.

### Quyết định

- Giữ `v13_INT8FFNProbe.py` làm accuracy-only negative ablation có thể tái lập.
- Không viết Triton/CUTLASS INT8 kernel, không chạy GPU performance và không
  thêm V13 vào scheduler vì numerical recipe đã fail canary trước performance.
- Không xem INT16 là bước tiếp theo: không có lợi thế Tensor Core/memory so với
  FP16 đủ để bù fixed-point scaling và conversion trong workload này.
- Chỉ mở lại INT8 nếu recipe khác về bản chất—ví dụ outlier routing,
  error-correction hoặc calibration/QAT—pass strict multi-seed accuracy probe
  trước. Groupwise K scaling còn phải chứng minh chi phí partial GEMM/reduction,
  không được suy speedup từ INT8 peak throughput.

### Hệ quả

- V11 tiếp tục là main; V12 vẫn là precision-margin ablation đang chờ GPU.
- Official #7/#10/#6/#8 không được chạy cho V13 sau khi #2 fail, đúng accuracy
  gate và tiết kiệm GPU/kernel-development time.
- V13 latency từ fake-quant PyTorch graph không phải INT8 performance evidence
  và không được đưa vào technical-report speedup table.

## D-029 — Dùng batch-chunk exact riêng cho shape #14, chưa promote vào main

**Trạng thái:** Accepted as validated candidate; promotion pending
**Ngày:** 2026-08-30

### Bối cảnh

Official #14 có input/output FP32 mỗi tensor `12.207 GiB`. V11 full-batch OOM
khi packed QKV FP16 cần thêm `18.311 GiB`; original reference còn materialize
score khoảng `18.6 TiB`. B=1 full-sequence V11 chạy được với peak `2.964 GiB`.

### Quyết định

- Tạo `v14_BatchChunked.py`: chỉ exact config #14 FP32 eval causal mới chạy
  từng batch sample qua đủ hai layer rồi ghi vào output preallocated.
- Không sửa baseline, workload, tolerance hoặc comparator. Accuracy dùng
  `shape14_accuracy.py`, giữ formula nhưng query-block reference để bounded memory.
- Chưa đổi `main.py`: scorer output lifetime, repeated-call cleanup và
  `torch.compile` behavior cần được kiểm tra trước promotion.
- Latency #14 chỉ báo optimized-only với baseline/speedup N/A; không suy speedup
  từ FLOP estimate hoặc từ reference query-blocked diagnostic.

### Hệ quả

- Full seed-1234 #14 PASS strict `0/3,276,800,000`, max abs `0.000831008`.
- Full forward peak `28.526 GiB`; optimized-only median `6683.9873 ms` qua năm
  CUDA Event repeats sau một warmup.
- Runner lặp phải giải phóng output `12.207 GiB`, chạy `gc.collect()` và cache
  cleanup ngoài vùng timing; nếu caller giữ output cũ thì output mới có thể OOM.
- Accuracy coverage đã có candidate cho đủ 14 shapes, nhưng promoted main claim
  vẫn #1–#13 cho tới khi D-029 được supersede bằng quyết định promotion.

## D-030 — Promote V14.1 với cutoff sequence-length làm main

**Trạng thái:** Accepted; supersedes promotion hold in D-029
**Ngày:** 2026-08-30

### Bối cảnh

V14 đã chứng minh batch-chunk size 1 giải được peak-memory của official #14 và
PASS strict một full seed. Project owner yêu cầu hợp nhất V11/V14 thành V14.1
và dùng cutoff theo `S`, thay vì chỉ so khớp toàn bộ tuple shape #14.

### Quyết định

- Tạo `v14_1_BatchChunked.py`, kế thừa nguyên V11 và dùng runtime cutoff
  `S >= 8192` cho FP32 eval khi `B > 1`; chunk size giữ là 1.
- Cutoff nằm cao hơn toàn bộ official #1–#13 (`S <= 1024`) và bắt official #14
  (`S=100000`). Dưới cutoff, training, non-FP32 hoặc `B=1` fallback trực tiếp
  V11, không thêm output copy.
- Tắt Dynamo capture riêng helper large-sequence để tránh unroll loop 32 batch
  thành graph khổng lồ; nhánh V11 thông thường vẫn cho phép `torch.compile`.
- Promote `main.py` và aliases `main`/`best` sang V14.1. Giữ V11 và V14 thành
  versioned rollback/ablation; không ghi đè evidence cũ.
- Không thay baseline, public contract, shape, tolerance, strict comparator hay
  arithmetic bên trong từng sample. #14 tiếp tục chỉ báo optimized latency với
  baseline/speedup N/A.

### Hệ quả

- Một artifact chính bao phủ dispatch cho cả 14 official shapes; #1–#13 giữ
  nguyên V11 path, #14 dùng memory-bounded execution schedule.
- Cutoff là routing policy, không phải tuyên bố breakpoint tối ưu cho mọi shape
  lạ. Hardware/config mới phải đo lại trước khi thay giá trị này.
- Caller vẫn phải giải phóng output full-size trước repeated #14 forward; giữ
  đồng thời hai output `12.207 GiB` có thể OOM dù internal chunking đúng.

## D-031 — Promote V15 direct-layout QKV cho exact shape #13

**Trạng thái:** Accepted; V15 supersedes V14.1 as the main wrapper
**Ngày:** 2026-08-30

### Bối cảnh

V14.1 dùng packed QKV `[B,S,3D]`; các Q/K/V view truyền vào Flash có sequence
stride `3D`. V15 thay riêng projection của official #13 bằng Triton GEMM ghi
trực tiếp `[3,B,H,S,Dh]`, tạo ba tensor contiguous mà không thêm transpose
kernel. Mọi shape và branch khác kế thừa nguyên V14.1, gồm batch-chunk #14.

### Quyết định

- Promote `main.py` sang `v15_DirectQKVLayout.py`; aliases `main`/`best` tiếp
  tục resolve qua stable entrypoint. Giữ V14.1 làm rollback.
- Chỉ bật direct-layout kernel cho exact official #13 causal FP32 eval:
  `B64/S1024/D128/H4/L4/FFN128`. Không ngoại suy schedule sang shape lạ.
- Giữ FP16 activation/weight/bias, FP32 dot accumulator và FP16 store trước
  Flash, nên precision boundary còn lại của V11/V14.1 không đổi.
- Chấp nhận promotion vì strict official accuracy PASS và paired end-to-end
  cùng raw-GPU latency giữ dấu ở cả hai implementation orders. Không dùng
  isolated GEMM hoặc kernel-count reduction làm tiêu chí thay thế.

### Evidence

- Official #13, năm trial: PASS `0/41,943,040`, max abs `0.00147235`.
- Paired dài `warmup=50/repeats=200/rounds=5`, V14.1→V15:
  `1.1080 → 1.0971 ms` (`-0.98%`), raw GPU `1.0966 → 1.0838 ms` (`-1.17%`).
- Reverse V15→V14.1: viết theo V14.1→V15 là `1.1251 → 1.1003 ms`
  (`-2.20%`), raw GPU `1.1062 → 1.0867 ms` (`-1.76%`).
- Topology giảm `21 → 17` Triton GPU events, trong khi Flash kernel time gần
  trung tính; gain đến từ QKV/projection-layout region.
- Exact #13 với 25% prefix padding PASS `0/25,165,824`; non-causal + padding
  diagnostic fallback PASS `0/13,056`. Official #2/#12 fallback canaries cũng
  PASS strict và giữ expected V14.1 latency class.

### Hệ quả

- Main scheduler hiện có ba vùng: V15 direct QKV cho exact #13, V14.1/V11 cho
  `S < 8192` còn lại, và batch chunk cho FP32 eval `S >= 8192` với `B > 1`.
- Gain #13 nhỏ (`0.98–2.20%`) và architecture-specific; GPU khác phải autotune
  và paired-rerun trước khi giữ dispatch.
- Evidence #14 không bị thay đổi vì V15 không vào direct-QKV branch ở #14.

## D-032 — Promote V16 compiled B=1 executor cho shape #14

**Trạng thái:** Accepted; V16 supersedes V15 as the main wrapper
**Ngày:** 2026-08-30

### Bối cảnh

V14.1 cố ý đặt cả loop batch-chunk #14 dưới `torch.compiler.disable` để tránh
Dynamo unroll 32 sample và phá memory bound. Hệ quả là thân Transformer B=1
cũng chạy eager 32 lần. V16 tách hai tầng: loop ngoài vẫn eager, còn cùng một
bound callable B=1 được compile một lần sau load/device/eval rồi tái sử dụng.

### Quyết định

- Promote `main.py` sang `v16_CompiledBatchExecutor.py`; giữ V15 làm QKV
  rollback và V14.1 làm eager large-sequence rollback.
- Chỉ dùng executor cho inherited large-sequence dispatch FP32 eval,
  `B>1`, `S>=8192`. Không thay arithmetic, chunk size, cutoff, public contract,
  tolerance, strict comparator hoặc workload.
- Xem compiled callable là derived inference cache: không đăng ký vào module
  state và invalidate sau `load_state_dict()`, `_apply()`/`.to()` hoặc đổi mode.
- Accuracy #14 phải gọi đúng compiled sample executor; compile/autotune nằm
  ngoài timed repeats. Original baseline/speedup tiếp tục N/A vì score ~18.6 TiB.

### Evidence

- Local forced-chunk: mask/no-mask bitwise-identical V15; state-dict/cache,
  training, BF16, short-sequence, Dynamo-eager và CPU Inductor gates PASS.
- RTX 5090/PyTorch `2.11.0+cu128`, full official #14: PASS
  `0/3,276,800,000`, max abs `0.000944197`, mean abs `6.56367e-05`, accuracy
  peak `19.585 GiB`.
- Cùng seed/TF32, warmup 1/repeats 5, sandwich V14.1→V16→V14.1:
  medians `7396.7202 → 7166.8359 → 7435.5688 ms`. V16 giảm `3.11–3.61%`,
  throughput `446,501 token/s`, timed peak `24.487 GiB` so với `26.977 GiB`.
- Official #2/#13 fallback canaries PASS strict, max abs lần lượt
  `0.000905275` và `0.00147235`; timing `1/1/1` bị loại.

### Hệ quả

- Main scheduler có bốn tầng kế thừa: V16 compiled sample executor cho #14,
  V15 direct QKV cho exact #13, V11/V14.1 path cho shape nhỏ còn lại và
  baseline-math fallback cho training/non-FP32.
- Gain và memory reduction mới được đo trên PyTorch 2.11/cu128; PyTorch 2.13
  chỉ có correctness canary. Phải rerun trước khi coi schedule portable.
- `/workspace` của Vast instance không persistent; evidence chính phải được
  ghi/sync về repository local, không phụ thuộc cache/artifact trên instance.

## D-033 — Giữ V17 batch-chunk B=2 làm ablation, không thay V16 main

**Trạng thái:** Accepted; V16 remains main
**Ngày:** 2026-08-30

### Bối cảnh

V16 compiled executor B=1 còn khoảng `8.1 GiB` headroom trên RTX 5090 32 GiB.
V17 thử gom hai sample #14 vào cùng compiled graph để giảm số executor calls từ
32 xuống 16 và tăng GEMM/Flash batch, không đổi số học hay precision boundary.

### Quyết định

- Giữ `v17_CompiledBatch2.py` và aliases `v17*` như versioned ablation.
- Không đổi `main.py`: V16 B=1 tiếp tục là promoted large-sequence path.
- Không thử B=4 trong lượt này. B=2 đã cho thấy batching tiếp chỉ tối ưu phần
  overhead nhỏ; muốn mở lại phải có profiler evidence hoặc measurement dài hơn
  chứng minh gain vượt drift.
- Giữ accuracy harness hỗ trợ executor batch group để future chunk-size
  candidate được strict-check đúng graph, không vô tình fallback về B=1.

### Evidence

- Local B2/B3 mask/no-mask bitwise-equivalent V16; cache lifecycle và
  training/BF16/short-sequence fallbacks PASS.
- Full official #14 strict PASS `0/3,276,800,000`, max abs `0.000944197`, mean
  abs `6.56366e-05`, accuracy peak `20.348 GiB`.
- Alternating V16→V17→V16→V17, warmup 1/repeats 5, cho medians
  `7183.8022 → 7141.3345 → 7162.9731 → 7131.5425 ms`. V17 nhanh hơn adjacent
  controls `0.30–0.59%`; trung bình hai medians giảm `0.515%`.
- Timed peak không đổi ở `24.487 GiB`. Official #2/#13 fallback canaries PASS;
  timing `1/1/1` của chúng bị loại.

### Hệ quả

- Chunk B=2 là numerically safe và memory-safe trên target, nhưng gain dưới 1%
  chưa đủ confidence để thêm một main specialization; từng process còn có
  thermal/clock drift theo repeat.
- V16 giữ implementation đơn giản hơn và evidence promotion mạnh hơn
  (`3.11–3.61%` so V14.1). V17 có thể được rerun bằng interleaved single-process
  harness hoặc profiler launch attribution nếu cần khai thác nốt khoảng 0.5%.

## D-034 — Reject V18 SageAttention; giữ PyTorch Flash cho shape #14

**Trạng thái:** Accepted; V16 remains main
**Ngày:** 2026-08-30

### Bối cảnh

Inner-executor profiler xác nhận attention chiếm `92.258%` raw device time của
V16 shape #14. PyTorch Flash thắng cuDNN/Efficient và FlashAttention-4 b28 trên
exact isolated workload. SageAttention PV-FP16/FP32-accum là external kernel duy
nhất cho isolated upside đáng kể (`1.3933x`), nhưng direct attention còn
`94/102.4M` strict violations.

### Quyết định

- Giữ `v18_SageAttentionShape14.py` làm negative accuracy/integration ablation.
- Không thêm SageAttention dependency vào stable submission và không đổi
  `main.py`; V16 tiếp tục dùng PyTorch Flash-first path.
- Không benchmark full-model V18 sau khi strict B=1 canary fail. Một external
  kernel chỉ được mở lại khi accuracy pass trước và compiled wrapper chứng minh
  eager/compiled equivalence bằng `torch.library.opcheck` cùng output compare.
- Giữ `shape14_profile.py`, FA4 probe và Sage probe làm reproducible selection
  harness; isolated kernel timing không được gọi là model speedup.

### Evidence

- Eager V18 official-config B=1, query-blocked reference: **FAIL
  `1/102,400,000`**, max abs `0.0026415`, mean abs `7.90562e-05`.
- Fresh-cache compiled wrapper fail integration `59,375,874/102,400,000`, max
  abs `1.08499`; đây không phải algorithm accuracy result.
- FA4 b28 strict attention PASS nhưng chậm hơn PyTorch Flash `7.72%`; Sage auto
  FP8 recipe fail accuracy mạnh; Sage PV-FP16 recipe nhanh nhưng vẫn không đạt
  strict gate.

### Hệ quả

- Target tiếp theo không nên chỉ đổi attention library. Cần một recipe khác về
  bản chất: selective exact correction, higher-precision QK, hoặc custom kernel
  giữ strict error budget; mọi phương án vẫn đi accuracy-first.
- `main.py` không nhận optional runtime dependency và rollback topology
  V16→V15→V14.1→V11 không đổi.

## D-035 — Không force direct-layout QKV toàn bộ; giữ #6 làm dispatch candidate

**Trạng thái:** Accepted; no main change
**Ngày:** 2026-08-30

### Bối cảnh

V15 direct-layout QKV thắng exact #13, nhưng branch ban đầu hardcode tuple
official #13. V15.1 được tạo để force cùng operator cho causal `S<8192` và đo
official #1–#12, nhằm tìm workload regime thay vì suy rộng từ một test.

### Quyết định

- Giữ `v15_1_DirectQKVAll.py` làm versioned cross-shape ablation; không đổi V15,
  V16 hoặc `main.py`.
- Reject unconditional direct-layout. Chỉ official #6 được giữ làm candidate
  cho một dispatcher tiếp theo; 11 shape còn lại dùng packed V15/V14.1 path.
- Nếu promote #6, predicate phải dựa trên workload có nghĩa như large `B*S`
  với `D=FFN=128`, không thêm một exact official-shape tuple không giải thích.
- Promotion cần robustness/aggregate rerun riêng; kết quả sweep này không tự
  cấp quyền thêm branch vào stable main.

### Evidence

- Official #1–#12 strict accuracy năm trial: PASS toàn bộ,
  `0/896,942,080` failed, max abs `0.00179085`.
- #6 giữ dấu ở cả hai orders: end-to-end `-2.41%/-4.44%`, raw device
  `-3.03%/-2.96%`; geometric delta `-3.43%/-2.99%`.
- #2/#4 apparent small wins bị retest `warmup=100,repeats=1000,rounds=7` loại:
  raw device lần lượt chậm `16.10–17.62%` và `5.52–6.17%`.
- #1/#3/#5/#7–#12 hoặc regress, đổi dấu, hoặc không có device corroboration.

### Hệ quả

- Lợi ích của direct-layout phụ thuộc consumer/layout amortization: batch-token
  volume lớn của #6 và sequence dài của #13 đủ bù custom projection/copy;
  shape nhỏ thì thêm bốn kernel mỗi forward thường bất lợi.
- Current test-based exact-#13 branch vẫn được giữ vì có evidence đã promote,
  nhưng sweep này cung cấp hướng tổng quát hóa hợp lý hơn cho lần sửa scheduler
  tiếp theo và tránh thêm branching theo test ID một cách mù quáng.

## D-036 — Thêm V16.1 làm control bỏ #13 nhưng giữ #14; không đổi main

**Trạng thái:** Superseded in implementation by D-037; historical control
**Ngày:** 2026-08-30

### Bối cảnh

V14.1 bỏ direct-QKV #13 nhưng đồng thời thiếu compiled executor #14 của V16,
nên không trả lời one-variable câu hỏi “current V16 nếu bỏ optimization #13”.

### Quyết định

- Thêm `v16_1_NoDirectQKV13.py`: kế thừa V16 và chỉ tắt construction flag
  `_use_direct_qkv_layout`.
- Giữ `main.py` trỏ V16. V16.1 là executable control, không phải promotion.
- Không rerun full #14 gate: exact #14 vốn không eligible direct-QKV ở cả
  V16/V16.1; executor implementation/cache/state đều kế thừa không đổi.

### Evidence

- Local exact-#13 config khớp V14.1 bitwise sau cùng weights; `state_dict`,
  training, alias, Dynamo-eager và #14 executor/flag gates PASS.
- RTX 5090 official #13 năm trial PASS `0/41,943,040`, max abs `0.00147235`.
- Paired hai orders có cùng graph (`29` kernels, `21` Triton events); V16.1
  chậm hơn V14.1 `0.16–0.76%` median, nằm trong control drift và không tạo
  performance claim.

### Hệ quả

- Khi cần ablate riêng lợi ích #13 mà vẫn giữ memory/performance path #14, dùng
  alias `v16.1`; khi cần stable submission tiếp tục dùng V16.
- Rollback hierarchy giờ có control composition rõ ràng thay vì phải chọn
  V14.1 và vô tình mất luôn optimization #14.

## D-037 — Source-clean V16.1 kế thừa V14.1 trực tiếp

**Trạng thái:** Accepted as control; V16 remains main
**Ngày:** 2026-08-30

### Bối cảnh

D-036 runtime-control vẫn đi qua MRO V16→V15: exact-#13 predicate được evaluate
rồi overwrite `False`. Dù graph sạch, source còn dead test-based condition.

### Quyết định

- Thay implementation V16.1 bằng class kế thừa trực tiếp V14.1 và chứa riêng
  validated compiled-executor/cache methods của V16.
- Không import/inherit V15 hoặc V16 trong V16.1; không giữ direct-QKV flag hay
  exact official tuple.
- Không refactor stable V16/main thành mixin trong lượt này để tránh mở rộng
  regression surface; duplication nhỏ trong control được chấp nhận và ghi rõ.
- Thêm `v16.1` vào `shape14_accuracy.py` để accuracy gate đúng executor mới.

### Evidence

- Source/MRO audit: V16.1→V14.1→V11→V8→V4.3→mixed→baseline; `rg` không thấy
  V15/V16 import, direct-QKV symbol hoặc exact config comparison.
- Local #13 và forced large-sequence/prefix-mask outputs khớp V14.1 bitwise;
  executor build/reuse/invalidation gates PASS.
- Official #13 năm trial PASS `0/41,943,040`, max abs `0.00147235`; paired
  graph vẫn `29` kernels/`21` Triton events, host delta đổi dấu
  `+0.93%/-0.98%` theo order.
- Exact-config #14 compiled B=1 canary PASS `0/102,400,000`, max abs
  `0.000719786`, peak `19.967 GiB`.

### Hệ quả

- Ngoài intentional large-sequence cutoff/#14 schedule, V16.1 không còn exact
  #13 tuple/direct-QKV branch ở source hay runtime. V8 workload predicate lịch
  sử vẫn nằm sâu trong parent source nhưng V11 override flag universal; nó
  không dispatch official cases trong V16.1.
- `main.py` vẫn V16; V16.1 là audit/control artifact dùng khi cần bỏ #13 nhưng
  giữ compiled #14 executor.

## D-038 — Promote source-clean V16.1 làm main

**Trạng thái:** Accepted; V16.1 supersedes V16 as the main wrapper
**Ngày:** 2026-08-31

### Bối cảnh

V16.1 đã tách hoàn toàn V15/direct-QKV khỏi source và MRO nhưng vẫn giữ reusable
compiled B=1 executor của V16 cho large-sequence path. Official #13 PASS strict
năm trial, graph khớp V14.1; exact-config #14 compiled B=1 canary cũng PASS.
Project owner chọn source topology tổng quát, không chứa exact official-#13 tuple,
làm artifact submission chính.

### Quyết định

- Promote `main.py` sang `v16_1_NoDirectQKV13.py`; aliases `main`/`best` và
  matrix-runner default tiếp tục resolve qua stable entrypoint này.
- Giữ V16 làm rollback để khôi phục direct-layout QKV #13 nếu ưu tiên latency
  hơn source cleanliness. V15/V14.1/V11 tiếp tục là các rollback thấp hơn.
- Không tuyên bố V16.1 nhanh hơn V16. Promotion này là lựa chọn kiến trúc:
  bỏ exact-test specialization và chấp nhận mất measured V15/V16 win
  `0.98–2.20%` ở official #13.
- Giữ nguyên compiled executor, large-sequence cutoff/chunk size, public API,
  state dict, workload, tolerance và strict comparator.

### Evidence

- Source/MRO audit: V16.1→V14.1→V11→V8→V4.3→mixed→baseline; không import V15/V16,
  không có direct-QKV flag hoặc exact official tuple.
- Official #13: PASS `0/41,943,040`, max abs `0.00147235`; paired V14.1/V16.1
  cùng `29` kernels/`21` Triton events và host delta đổi dấu `+0.93%/-0.98%`.
- Exact-config #14 compiled B=1 canary: PASS `0/102,400,000`, max abs
  `0.000719786`, peak `19.967 GiB`.
- Full #14 V16 evidence `0/3,276,800,000` và `3.11–3.61%` gain so V14.1 vẫn
  chứng minh schedule được copy, nhưng full V16.1 #14 rerun còn là validation
  cần làm trước final submission; không suy canary thành một full-run result.

### Hệ quả

- Stable main không còn direct-QKV optimization ở #13; #1–#13 dùng packed-QKV
  V14.1/V11 path, còn large-sequence path dùng standalone V16-equivalent
  compiled executor.
- Candidate direct-QKV tiếp theo, nếu có, phải bắt đầu từ V16.1 và dùng workload
  predicate có ý nghĩa như large `B*S`, không quay lại hard-code test tuple.
- Technical report phải tách rõ promotion vì source cleanliness khỏi performance
  promotion và tiếp tục ghi V16 full-#14/V15 #13 measurements như predecessor
  evidence, không gán lại thành số đo mới của V16.1.

## D-039 — Mở lại Sage bằng exact-prefix correction; không đổi main trước gate

**Trạng thái:** Accepted for experimentation; promotion pending
**Ngày:** 2026-08-31

### Bối cảnh

V18 raw Sage bị reject vì full-model B=1 còn một strict violation và custom op
cho output sai dưới CUDA Graph. Failure-locality artifact sau đó cho exact #14
seed 1234 thấy 109 isolated attention violations đều nằm ở query `1..31`, với
`failed_outside_prefix[32] == 0`. Project owner yêu cầu code candidate
`v17_sage` để quét toàn bộ official suite và ưu tiên đưa sai số về chuẩn.

### Quyết định

- Tạo `v17_sage.py` trên source-clean V16.1; tên report là V17-Sage để phân biệt
  với historical V17-B2. Alias `v17` vẫn trỏ `v17_CompiledBatch2.py`; candidate
  mới dùng `v17.sage`/`v17_sage`.
- Sage path dùng per-thread INT8-QK, PV-FP16/FP32-accum, exact causal prefix 32
  và FP32 attention out-projection. Không nới tolerance hoặc đổi workload.
- Dispatch theo support envelope `causal`, FP32 eval, `S>32`, `head_dim<=128`;
  branch không hỗ trợ fallback V16.1. Official run phải bật dependency guard để
  không ghi fallback như một Sage result.
- Đánh dấu custom op CUDA-Graph-unsafe và gate bằng `torch.library.opcheck`
  cùng eager/compiled no-CUDA-Graph equivalence trước model matrix.
- Không đổi `main.py` và không benchmark shape chưa PASS strict accuracy.

### Hệ quả

- Full #1–#13 matrix có thể audit cả regression và support fallback; #14 tiếp
  tục dùng memory-bounded accuracy tool B=1→2→32 trước optimized-only timing.
- Local CPU/fake gates không chứng minh Sage arithmetic. Candidate chỉ có model
  result sau khi chạy trên target có đúng SageAttention 2.2.0/source pin.
- Nếu prefix 32 không robust qua seed/shape, chỉ tăng prefix theo measured
  locality hoặc reject candidate; không suy một seed thành universal accuracy.

## D-040 — Reject V17-Sage sau full #1–#13; giữ V16.1 làm main

**Trạng thái:** Accepted
**Ngày:** 2026-08-31

### Bối cảnh

V17-Sage đã chạy official #1–#13 trên RTX 5090/PyTorch `2.11.0+cu128`, đúng
SageAttention 2.2.0, accuracy trials `5` và benchmark `20/100/3`, optimized
compile `max-autotune`. Exact-prefix 32 cùng FP32 attention out-projection vẫn
không đủ robust: #6 fail với max abs `0.00250164`, #9 fail `0.00255397`.

### Quyết định

- Reject V17-Sage khỏi promotion và giữ file/aliases làm negative ablation.
- Giữ `main.py` trỏ source-clean V16.1. Không tăng tolerance, bỏ case hoặc dùng
  #13 speedup `17.067x` để che hai official accuracy failures.
- #8/#12 numbers của run là V16.1 fallback, không được gán cho Sage.

### Hệ quả

- Correctness vẫn là gate của mọi submission candidate.
- Selective Sage router chỉ được xem lại nếu có recipe/correction mới pass đầy
  đủ các shape được bật; một kết quả tốt ở long sequence không đủ để promote.

## D-041 — Cho phép V18-Sage direct automatic như performance-only probe

**Trạng thái:** Accepted for diagnostic only; not promotion-eligible
**Ngày:** 2026-08-31

### Bối cảnh

Owner yêu cầu đo raw Sage không correction và chủ đích chưa quan tâm numerical
accuracy. Mục tiêu là tách overhead của V17 exact-prefix/FP32-out khỏi backend
automatic SM120, không phải tạo một accuracy-valid submission.

### Quyết định

- Tạo `v18_sage.py` kế thừa trực tiếp V16.1 và gọi `sageattention.sageattn`.
  Trên source pin/SM120, automatic dispatch là INT8-QK per-warp + FP8-PV
  `fp32+fp16`; không exact-prefix và không FP32 attention out-projection.
- Bật cho causal FP32 CUDA eval với `head_dim<=128`; #8 Dh=256 fallback V16.1.
  Alias `v18.sage`/`v18_sage` không thay historical alias `v18`.
- Matrix runner được phép forward opt-in `--benchmark-on-failure` đã có trong
  official harness. Comparator/tolerance không đổi, status vẫn
  `ACCURACY_FAIL`, và timing của row fail phải ghi là invalid diagnostic.
- `main.py` tiếp tục là V16.1. Không chạy full benchmark trong implementation
  turn; owner chạy target matrix sau khi light custom-op/integration gate pass.

### Hệ quả

- V18-Sage có thể cung cấp trần performance theo shape nhưng không thể được
  promote nếu chưa pass lại toàn bộ strict accuracy suite.
- `cudagraph_unsafe` và no-CUDA-Graph shape-#14 executor vẫn bắt buộc để tránh
  đo output replay sai như một speedup.

## D-042 — Flatten V16.1 thành artifact standalone và archive version cũ

**Trạng thái:** Accepted; packaging/source-topology change, no algorithm change
**Ngày:** 2026-08-31

### Bối cảnh

V16.1 đã bỏ V15/direct-QKV nhưng source active vẫn kế thừa chuỗi
V14.1→V11→V8→V4.3→mixed→baseline. Repository root vì thế chứa nhiều version
và artifact submission không thể dùng độc lập ngoài repository.

### Quyết định

- Tạo `v16_1_clean.py` chứa trọn config/model, reference fallback, FP16 cache,
  Flash-first attention, Triton FFN-in/FP32-pre-GELU và compiled executor #14.
- File clean chỉ import PyTorch và optional Triton; không import benchmark
  harness hoặc implementation `v*.py` khác.
- `main.py`, matrix/profile runner và các tool shape #14 dùng file clean.
- Chuyển toàn bộ 35 file version lịch sử sang `archive/versions/`; chúng không
  còn là active aliases nhưng vẫn được giữ nguyên làm evidence/rollback.
- Không thay baseline, public API, state dict, arithmetic, cutoff, compile mode,
  comparator, tolerance hoặc performance claims.

### Evidence

- Strict state-dict load giữa composed V16.1 cũ và clean PASS.
- CPU eval causal/non-causal × mask/no-mask khớp bit-for-bit.
- Training FP32/BF16 fallback khớp bit-for-bit.
- Forced large-sequence compiled-eager executor khớp bit-for-bit; cache reuse
  và invalidation sau `load_state_dict()`/`.to()` PASS.

### Hệ quả

- Root chỉ còn một implementation version: `v16_1_clean.py`.
- Historical commands cần dùng file trong `archive/versions/`; số liệu cũ vẫn
  giữ nguyên và không được gán thành benchmark mới của artifact clean.
- Tại thời điểm D-042, CUDA/Triton equivalence và full official GPU matrix vẫn
  phải chạy trước final submission; D-043 bên dưới đóng gate này.

## D-043 — Chốt standalone V16.1 làm artifact và evidence nộp cuối

**Trạng thái:** Accepted; final official validation complete
**Ngày:** 2026-08-31

### Bối cảnh

D-042 chỉ chứng minh equivalence local sau flatten. Để gán correctness và
performance cho đúng artifact nộp, cần fresh GPU run trên `main.py →
v16_1_clean.py`, không kế thừa số V4.3/V16 lịch sử.

### Quyết định

- Chốt commit `4f77a04fd51c3a2ad8b9d8986657915ae3ca94d6` làm revision của final
  evidence.
- Dùng RTX 5090 `sm120`, PyTorch `2.11.0+cu128`, Triton `3.6.0`; #1–#13 chạy
  strict accuracy năm trial và paired timing `20/100/3` với optimized
  `max-autotune`.
- Shape #14 chạy full streaming strict gate đủ `B=32`; performance chỉ báo
  optimized-only vì original reference cần score khoảng `18.6 TiB`.
- Check in curated raw JSON/CSV/log và environment manifest dưới
  `results/final/`; không commit profiler dump lớn hoặc credential/endpoint.

### Evidence

- #1–#13 PASS toàn bộ, failed `0`, worst max abs `0.00179085`, geomean speedup
  `7.904x`; per-shape range `2.489x–33.925x`.
- #14 PASS `0/3,276,800,000`, max abs `0.000944197`, mean abs
  `6.56367e-05`.
- #14 optimized-only median `7213.5254 ms`, p90 `7252.4406 ms`, throughput
  `443,611.11 token/s`, peak `24.487 GiB`; baseline/speedup N/A.

### Hệ quả

- Technical report được phép dùng final active-main numbers trên làm headline.
- V4.3/V15/V16/V17 numbers chỉ còn là ablation/predecessor evidence.
- Gate còn mở chỉ là robustness/portability nhiều seed, input scale, padding,
  hardware và cold-start; chúng không phủ định official single-seed final run.

## D-044 — Giữ V19 CUDA checkpointed-FP16 ngoài main cho tới GPU gate

**Trạng thái:** Rejected for promotion after GPU gate; V16.1 remains main
**Ngày:** 2026-08-31

### Bối cảnh

Owner yêu cầu thử FP16 accumulation bằng kernel CUDA thật nhưng promote partial
sum sang FP32 sau mỗi K=32, đồng thời chuẩn bị thêm các khoảng checkpoint để
sweep. V5.1 trước đây bật full FP16 accumulation process-global, fail strict
accuracy và không tạo paired speedup; V19 phải cô lập scope và accumulation
policy thay vì lặp lại global flag.

### Quyết định

- Tạo `v19_CUDAFP16Checkpoint.py` trên standalone V16.1 và chỉ thay
  `FFN-in -> exact GELU` bằng CUDA WMMA. Default checkpoint K=32; K=16/64/128
  và FP32 control giữ cùng block/layout/epilogue.
- Không bật `allow_fp16_accumulation`; QKV, attention, FFN-out,
  residual/LayerNorm, cache/state dict và executor #14 giữ nguyên V16.1.
- CUDA extension phải build trước `torch.compile`; lỗi build mặc định dừng run.
  Fallback chỉ được bật explicit và timing đó không phải V19 evidence.
- Không đổi `main.py` hoặc final evidence. V19 chỉ được cân nhắc sau strict
  CUDA canaries, #1–#13, full #14 và paired V16.1/V19 trên GPU idle.

### Evidence hiện tại

- Local CPU PyTorch 2.12.1: state dict/fallback/branch tests, custom-op
  `opcheck` và compile-eager smoke PASS.
- Official-shape-#2 one-trial portable simulation PASS cho K=16/32/64/128 và
  FP32 control; K=32 max abs `0.00108075`, failed `0/16,384`.
- RTX 5090/driver 595: mọi K strict PASS #7/#10/#2; K64 là FP16 mode nhanh
  nhất trên #6 nhưng `29.7675 ms` so V16.1 controls `25.1593/25.2380 ms`.
- K64 full #1–#13 PASS, geomean `10.3079x` so V16.1 `11.8030x`; direct
  optimized-latency geomean regress `13.73%`. Full #14 PASS nhưng two-order
  medians `7251.4170/7310.5811 ms`.

### Hệ quả

- V16.1 tiếp tục là stable artifact. V19 không được gán headline result hoặc
  gọi là nhanh hơn; measured GPU evidence đã reject promotion.
- Sweep sau phải tách accuracy của checkpoint policy khỏi chất lượng schedule
  CUDA bằng FP32 control cùng source; nếu control đã chậm V16.1, tối ưu tiling
  trước khi kết luận accumulation policy không có lợi.

## D-045 — Giữ V19.1 multi-stream scheduler ngoài main tới khi đóng memory gate

**Trạng thái:** GPU tuning complete; V19.1.0 P4 measured winner, main unchanged
**Ngày:** 2026-08-31

### Bối cảnh

Outer loop shape #14 của V16.1 chạy 32 executor B=1 tuần tự. Batch samples độc
lập nên có thể enqueue lên nhiều CUDA stream, nhưng concurrent intermediates có
thể vượt memory và một compiled CUDA Graph không an toàn khi replay đồng thời
với static buffers.

### Quyết định

- Tạo V19.1.0 từ V16.1 và V19.1.1 từ V19; cả hai chỉ ghép cùng outer scheduler,
  không đổi arithmetic/state dict của parent.
- Cho phép parts `1/2/4/8/16/32`, mặc định 2. Chia range liên tục cân bằng; mỗi
  worker stream xử lý tuần tự các sample B=1 trong range.
- Parts>1 bắt buộc inner Inductor mode `max-autotune-no-cudagraphs`. Stream cache
  là runtime-only và bị xóa cùng executor cache sau load/move/config change.
- `shape14_accuracy.py` gọi candidate theo group bằng số parts khi B>1 để
  correctness gate đi qua outer scheduler nhưng không giữ full B32 output trong
  lúc dựng memory-bounded reference.
- Không đổi `main.py`. Tăng parts tuần tự trên GPU idle và dừng tại OOM/memory
  headroom không an toàn; latency chỉ hợp lệ sau full strict #14 PASS.

### Evidence hiện tại

- Local planner cover B lẻ/chẵn cho mọi parts; parse/mode/state-dict và
  CPU parent-equivalence mask/no-mask, training, BF16 đều PASS.
- Official #2 one-trial smoke: V19.1.0 max abs `0.00084424`, V19.1.1 K=32
  portable max abs `0.00108075`, cả hai failed `0/16,384`.
- RTX 5090 P2/P4/P8 canaries execute multi-stream path thật và strict PASS.
  P8 regress, resident memory khoảng `29.6 GiB`; P16 không chạy.
- V19.1.0 P4 full #14 PASS `0/3.2768B`, max abs `0.000944138`; post-gate
  median `6780.3867 ms`, p90 `6792.4046 ms`, peak `25.676 GiB`. P4 thắng P1
  controls `1.51–1.66%`.
- V19.1.1 K64/P2 full PASS nhưng two-order average `7179.4322 ms`; V19.1.0
  P4 là overall winner. Raw evidence ở
  `results/v19-tune-rtx5090-driver595-20260901/`.

### Hệ quả

- Paired comparisons phải là V16.1↔V19.1.0 và V19↔V19.1.1, cùng
  no-CUDA-Graph policy, seed, warmup/repeats và GPU idle.
- Mỗi latency row phải kèm parts và peak allocation; theoretical overlap không
  được ghi thành speedup đã đo.
- Không tự động đổi `main.py`: promotion là quyết định riêng của owner sau khi
  cân nhắc gain #14 nhỏ và headroom của P4.

## D-046 — Promote driver-595 timeline sweep; giữ driver-580 làm cross-host archive

**Trạng thái:** Accepted; reporting/evidence promotion, no implementation change
**Ngày:** 2026-08-31

### Bối cảnh

Final evidence D-043 chỉ đo standalone V16.1 trên một host driver `580.159.03`.
Để báo cáo tiến trình phát triển trên cùng environment, cần rerun đủ #1–#13 cho
Baseline, V1, V2, V3.1 eager/compiled, V4.1, V4.2, V4.3, V8, V11 và V16.1.
Host mới dùng cùng RTX 5090/PyTorch/CUDA stack nhưng driver `595.71.05`, CPU và
RAM khác; vì vậy không được ghép latency hai host như một ablation code.

### Quyết định

- Promote V16.1 **start-control** của sweep driver-595 làm headline, không chọn
  end-control hoặc reverse run có score đẹp nhất.
- Chấp nhận sweep chỉ khi start/end baseline/optimized geomean và heavy shapes
  #6/#8/#13 drift không quá 3%.
- Mọi checkpoint chạy đủ #1–#13. V3.1 compiled strict FAIL phải giữ status và
  không có timing; không bật `--benchmark-on-failure`.
- Chạy reverse-order full matrices cho các aggregate difference dưới 3%.
- Theo scope cuối của owner, shape #14 chỉ có Baseline static-infeasible và
  V16.1 B1/streamed/native/timing; không report #14 cho checkpoint lịch sử.
- Promote artifacts vào `results/final/` và
  `results/timeline-rtx5090-driver595/`; giữ D-043 evidence nguyên vẹn trong
  `results/cross-host-driver580/`.

### Evidence

- V16.1 start/end geomean `11.8030x` / `11.8383x`; baseline geomean drift
  `0.166%`, optimized `0.458%`; max heavy-shape drift `1.042%`. Gate PASS.
- Baseline, V1, V2, V3.1 eager, V4.1, V4.2, V4.3, V8, V11 và hai V16.1
  controls PASS 13/13. V3.1 compiled fail 13/13, tổng `201,682` failed elements,
  timing skipped.
- Forward geomeans: `1.0011/1.0763/1.4353/2.1006/N/A/10.1999/10.4489/
  11.6755/11.7854/11.7483/11.8030x` theo checkpoint order.
- #14 V16.1 streamed strict PASS `0/3,276,800,000`, native B32 PASS; median
  `6987.4644 ms`, p90 `6994.0999 ms`, throughput `457,962.98 token/s`, peak
  `24.487 GiB`. Baseline latency/speedup N/A.

### Hệ quả

- Headline hiện tại là `11.803x` trên environment driver-595; V16.1 vẫn là main
  và source revision không đổi.
- Driver-595 baseline geomean chậm hơn driver-580 `72.48%`; optimized geomean
  chậm hơn `15.51%`. Chênh `7.904x → 11.803x` là cross-host ratio effect, không
  phải code improvement.
- Mọi so sánh version trong timeline dùng cùng driver-595 host/protocol; số
  D-043 chỉ dùng làm cross-host audit.
