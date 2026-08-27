# Kế hoạch triển khai

## 1. Trạng thái tổng quan

**Phase hiện tại:** Phase 2 — chạy accuracy matrix cho baseline/`v1` trên GPU mục tiêu.

Repository đã có đề bài, benchmark Torch chính thức cập nhật ngày 2026-08-27 và implementation `v1`. GPU vật lý index `1` đã được xác nhận là RTX 5090; quyền device và venv riêng của Track 3 đã hoạt động. Một smoke benchmark GPU đã pass, nhưng full 14-shape accuracy/performance matrix chưa hoàn tất.

## 2. Quy ước trạng thái

- `[x]` Đã hoàn thành và có artifact/code.
- `[~]` Đang làm hoặc mới hoàn thành một phần.
- `[ ]` Chưa bắt đầu.
- `[!]` Bị chặn bởi thông tin/môi trường bên ngoài.

## 3. Phase 0 — Nắm đề và dựng khung tài liệu

- [x] Lưu Track 3 vào `STATEMENT.md`.
- [x] Lưu tên hai benchmark resource từ Lark.
- [x] Thêm `AGENTS.md`, `ARCHITECTURE.md`, `SOLUTION.md`, `SOLUTIONS.md`, `IMPLEMENTATION_PLAN.md`, `DECISION.md`.
- [x] Cấu hình public repository: `https://github.com/wheres-my-perry/techjam-2026-track3`.
- [x] Đồng bộ correctness mặc định trong code với luật chính thức: `rtol=0.02`, `atol=0.002`, strict `<`.
- [x] Import attachment Torch ngày 2026-08-27 (source SHA-256 `5529c96a80799b51f68092e1444a30b17994554dffdf52da98ba701489a7f36e`) và chuẩn hóa comparator theo trang đề: strict `<`, `rtol=0.02`, `atol=0.002`.

**Exit criteria:** đề, resource và luật correctness được đối chiếu với bản phát hành chính thức.

## 4. Phase 1 — Baseline và harness

- [x] Có reference Transformer block và model.
- [x] Có random input generation với seed và padding mask.
- [x] Có elementwise accuracy check.
- [x] Có CUDA Event/CPU timing, warmup và alternating measurement order.
- [x] Có CLI cho shape, dtype, mask, tolerance, compile và TF32.
- [ ] Thêm xuất kết quả máy đọc được (JSON/CSV).
- [ ] Thêm script chạy benchmark matrix.
- [ ] Thêm metadata môi trường: GPU capability, driver, CUDA, cuDNN, OS và git revision.

**Exit criteria:** một command có thể chạy toàn bộ shape matrix và tạo artifact kết quả tái lập được.

## 5. Phase 2 — Validate `v1`

- [x] Implement packed QKV projection.
- [x] Refresh packed weights sau `load_state_dict()`.
- [x] Giữ nguyên attention math; fallback về projection gốc khi training.
- [x] Xác định GPU vật lý index `1` là NVIDIA GeForce RTX 5090.
- [x] Cấp quyền GPU cho tài khoản chạy task qua group `gpu`.
- [x] Tạo `/home/chim/techjam-2026-track3/.venv` với PyTorch `2.13.0+cu130` và CUDA `13.0`.
- [x] Upload workspace tới `/home/chim/techjam-2026-track3`.
- [x] Chạy syntax và CPU correctness smoke test bằng PyTorch `2.13.0+cu130` (PASS).
- [x] Validate fused QKV local trên FP32 causal/non-causal, padding và BF16 (PASS, exact output).
- [x] Chạy smoke test trong môi trường GPU sau khi được cấp quyền.
- [ ] Chạy accuracy matrix cho causal/non-causal và padding/no-padding.
- [ ] Chạy accuracy trên nhiều seed, input scale và tất cả shape công bố.
- [ ] Chạy benchmark GPU với warmup và nhiều round.
- [ ] Ghi kết quả, command và môi trường vào `SOLUTIONS.md`.
- [ ] Xác nhận packed cache không stale sau mọi đường load/move/compile được dùng.

**Exit criteria:** mọi case mục tiêu pass accuracy và có kết quả speedup tái lập trên GPU mục tiêu.

## 6. Phase 3 — Profile và chọn bottleneck

- [ ] Profile baseline và `v1` bằng PyTorch Profiler/Nsight.
- [ ] Đếm kernel launch theo block.
- [ ] Tách thời gian QKV, attention, output projection, LayerNorm và FFN.
- [ ] Đo peak memory/intermediate allocation.
- [ ] Xác định bottleneck theo từng nhóm shape.

**Exit criteria:** mỗi optimization tiếp theo có profiler evidence và target metric rõ ràng.

## 7. Phase 4 — Mở rộng optimization

- [x] Flatten `v2_SPDA.py` theo whole-model loop lịch sử; GPU PASS và đạt 1.781x non-causal, 1.922x official shape #1 causal.
- [x] Dùng bản `v1_old` tạm thời để đối chiếu rồi xóa sau khi đã merge logic tốt hơn vào `v2_SPDA.py`.
- [ ] Thử `torch.compile` với các mode và shape ổn định.
- [ ] Xây low-precision optimized path thay vì fallback toàn phần.
- [ ] Thử residual + LayerNorm fusion.
- [ ] Thử FFN/GELU fusion hoặc giảm intermediate allocation.
- [ ] Chỉ thử Triton/custom CUDA attention khi SDPA là bottleneck đã được xác nhận.
- [ ] So sánh code complexity, compile time, portability và speedup.

**Exit criteria:** có ít nhất một candidate tốt nhất cho mỗi nhóm shape quan trọng.

## 8. Phase 5 — Scheduler theo shape

- [ ] Chốt dispatch key.
- [ ] Xây registry của candidate đã pass correctness.
- [ ] Chạy offline autotuning trên GPU mục tiêu.
- [ ] Sinh bảng dispatch tĩnh và safe fallback.
- [ ] Kiểm tra scheduler overhead.
- [ ] Chạy lại full accuracy/performance matrix end-to-end.

**Exit criteria:** scheduler không làm sai case lạ và cải thiện aggregate score so với một implementation duy nhất.

## 9. Phase 6 — Submission

- [ ] Dọn repository và thêm `.gitignore`/dependency setup phù hợp.
- [ ] Viết README cài đặt, chạy và tái lập kết quả.
- [x] Chốt public repository.
- [~] Đã viết `SOLUTION.md` cho các implementation hiện tại; cần hoàn thiện bằng full matrix và profiler evidence.
- [ ] Chuẩn bị Devpost description.
- [ ] Quay demo video public trên YouTube.
- [ ] Kiểm tra licensing/trademark/copyright.

**Exit criteria:** một reviewer mới có thể clone, chạy accuracy/benchmark và tái lập bảng kết quả.

## 10. Việc ưu tiên tiếp theo

1. Chạy `v1_fuseQKV.py` trên đủ 14 test shapes chính thức, ưu tiên correctness trước performance.
2. Chạy causal × padding matrix và nhiều seed cho các shape khả thi.
3. Ghi kết quả vào `SOLUTIONS.md`, sau đó thêm JSON output/matrix runner.
4. Profile bottleneck theo nhóm shape trước khi triển khai candidate tiếp theo.
