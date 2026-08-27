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

Giữ candidate ở các file `vN.py` hoặc module có version rõ ràng cho đến khi có dữ liệu để promote. Mỗi version phải có entry trong `SOLUTIONS.md`.

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
