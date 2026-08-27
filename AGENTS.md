# Quy tắc làm việc trong repository

## Mục tiêu

Repository này phục vụ TikTok TechJam 2026, Track 3: tối ưu Transformer layer trên GPU mà vẫn đạt yêu cầu sai số trong [STATEMENT.md](STATEMENT.md).

Public repository: [wheres-my-perry/techjam-2026-track3](https://github.com/wheres-my-perry/techjam-2026-track3).

## Tài liệu cần đọc

Trước khi sửa code, đọc các tài liệu theo thứ tự sau:

1. [STATEMENT.md](STATEMENT.md) — đề bài và tiêu chí chấm điểm.
2. [ARCHITECTURE.md](ARCHITECTURE.md) — kiến trúc benchmark và luồng thực thi.
3. [DECISION.md](DECISION.md) — các quyết định kỹ thuật đã chốt.
4. [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) — phase hiện tại và việc tiếp theo.
5. [SOLUTIONS.md](SOLUTIONS.md) — các phương án đã thử hoặc đang cân nhắc.
6. [SOLUTION.md](SOLUTION.md) — technical report đầy đủ của các implementation hiện tại.

## Quy tắc bắt buộc

- Xem `torch_transformer_benchmark.py` là baseline/reference. Không sửa baseline chỉ để tạo speedup hoặc làm bài kiểm tra dễ hơn.
- Giữ nguyên giao diện `UserOptimizedTransformer.forward(x, valid_token_mask)` và shape đầu ra `[batch_size, seq_len, d_model]`.
- Không nới `rtol`, `atol`, bỏ test case, giảm workload hoặc đổi dữ liệu đầu vào để tuyên bố một phương án nhanh hơn.
- Correctness mặc định phải dùng đúng luật cuộc thi: `relative error < 0.02 OR absolute error < 0.002` (strict `<`, không phải `<=`).
- Correctness là cổng bắt buộc: chỉ benchmark hiệu năng sau khi accuracy pass, trừ khi đang debug và ghi rõ kết quả không hợp lệ.
- So sánh baseline và optimized trên cùng máy, GPU, dtype, shape, seed, warmup, số lần lặp và cấu hình TF32/compile.
- Mọi con số hiệu năng phải kèm command, môi trường, latency baseline, latency optimized và speedup. Không ghi kết quả ước lượng như kết quả đã đo.
- Tối ưu mới nên được lưu thành phiên bản riêng (`v2.py`, `v3.py`, ...) cho đến khi có quyết định promote. Không ghi đè một phiên bản đã có kết quả hữu ích.
- Nếu đổi tên parameter hoặc cấu trúc `state_dict`, cập nhật `copy_model_weights()` và thêm kiểm tra tương đương trọng số.
- Cache suy luận như packed weights phải được refresh sau `load_state_dict()` và không được dùng sai trong training.
- Luôn kiểm tra các nhánh: causal/non-causal, có/không padding mask, các dtype được hỗ trợ và nhiều input shape.
- Mỗi khi thay đổi thuật toán hoặc thêm solution mới, cập nhật `SOLUTION.md` để report luôn khớp code, correctness, benchmark và giới hạn hiện tại.
- Không commit `__pycache__`, file tạm, profiler dump lớn hoặc dữ liệu benchmark sinh tự động nếu chưa có chủ đích lưu trữ.

## Quy trình thay đổi

1. Ghi giả thuyết và phạm vi phương án vào `SOLUTIONS.md`.
2. Tạo hoặc cập nhật implementation phiên bản hóa.
3. Chạy accuracy matrix trước.
4. Chạy benchmark có warmup trên GPU mục tiêu.
5. Ghi command và kết quả vào `SOLUTIONS.md`.
6. Cập nhật thuật toán, kết quả và giới hạn trong `SOLUTION.md`.
7. Cập nhật trạng thái trong `IMPLEMENTATION_PLAN.md`.
8. Nếu thay đổi hướng kiến trúc hoặc đánh đổi dài hạn, thêm quyết định vào `DECISION.md`.

## Kiểm tra tối thiểu

```bash
python3 -m py_compile torch_transformer_benchmark.py v1_fuseQKV.py v2_SPDA.py
CUDA_VISIBLE_DEVICES=1 python3 v2_SPDA.py --device cuda:0 --dtype float32
```

GPU vật lý được chỉ định cho dự án là index `1`. Sau khi đặt `CUDA_VISIBLE_DEVICES=1`, process chỉ thấy GPU đó và PyTorch gọi nó là `cuda:0`; không đổi command thành `cuda:1` trong trường hợp này.

Trước khi kết luận phương án hoàn tất, chạy thêm ma trận shape/dtype/mask phù hợp với GPU mục tiêu. Một smoke test nhỏ không thay thế benchmark chính thức.
