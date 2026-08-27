# Kiến trúc repository

## 1. Phạm vi

Repository xây dựng một harness để so sánh Transformer reference với implementation tối ưu. Mục tiêu là giảm inference latency trên GPU mà không vượt ngưỡng sai số của đề bài.

Public repository: [wheres-my-perry/techjam-2026-track3](https://github.com/wheres-my-perry/techjam-2026-track3).

## 2. Bản đồ repository

| Thành phần | Vai trò |
|---|---|
| `STATEMENT.md` | Bản lưu đề Track 3, tài nguyên, deliverables và tiêu chí chấm điểm. |
| `torch_transformer_benchmark.py` | Reference implementation và benchmark harness gốc. |
| `v1_fuseQKV.py` | Phương án tối ưu đầu tiên: gộp Q/K/V projection thành một phép `F.linear`. |
| `v2_SPDA.py` | Candidate kết hợp packed-QKV view không-copy và PyTorch SDPA cho FP32. |
| `AGENTS.md` | Quy tắc làm việc cho người và coding agent. |
| `SOLUTION.md` | Technical report đầy đủ cho các implementation hiện tại. |
| `SOLUTIONS.md` | Danh mục phương án, thử nghiệm và kết quả. |
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

## 8. Kiến trúc `v1_fuseQKV.py`

`v1_fuseQKV.py` giữ nguyên cấu trúc parameter của baseline và thêm cache không persistent:

- Gộp Q/K/V weight và bias thành một packed projection.
- Refresh packed weights sau `load_state_dict()`.
- Chạy một `F.linear` rồi tách kết quả thành Q, K, V.
- Giữ nguyên toàn bộ attention math, mask và FFN của baseline.
- Khi training, dùng lại ba projection gốc để tránh cache stale.

Repository hiện có candidate SDPA; chưa có custom CUDA/Triton kernel hoặc scheduler theo shape.

## 9. Kiến trúc `v2_SPDA.py`

`v2_SPDA.py` giữ state dict của baseline, pack QKV thành một projection và thay attention core FP32 bằng `F.scaled_dot_product_attention`. Attention được inline trong whole-model loop để tránh module dispatch mỗi layer. Causal-only dùng `is_causal`; causal kết hợp padding dùng boolean mask. Packed weights được refresh sau `load_state_dict()`. FP16/BF16 fallback về reference cho đến khi low-precision accuracy được giải quyết.

## 10. Điểm mở rộng

- Shape scheduler chọn implementation theo `(B, S, D, H, dtype, causal, padding)`.
- Custom Triton/CUDA cho LayerNorm, QKV projection, attention hoặc FFN fusion.
- Autotuning và cache cấu hình theo GPU architecture.
- Benchmark matrix runner và xuất kết quả JSON/CSV.
- Profiler integration cho CUDA time, memory traffic và kernel launch count.
- Visualizer để so sánh latency/speedup theo shape sau khi có dữ liệu chuẩn hóa.

## 11. Môi trường GPU mục tiêu

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
CUDA_VISIBLE_DEVICES=1 python v1_fuseQKV.py --device cuda:0 --dtype float32
```

Sau khi mask bằng `CUDA_VISIBLE_DEVICES=1`, `cuda:0` bên trong process chính là GPU vật lý index `1`.

Tài khoản `chim` hiện thuộc group `gpu`. PyTorch trong venv Track 3 nhận đúng một RTX 5090 khi chạy với `CUDA_VISIBLE_DEVICES=1`; device logic bên trong process là `cuda:0`.
