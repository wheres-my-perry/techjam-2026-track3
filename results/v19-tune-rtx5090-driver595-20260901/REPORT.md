# V19/V19.1 RTX 5090 tuning report — 2026-09-01

## Kết luận

Winner là **V19.1.0 với `TECHJAM_V19_PARALLEL_PARTS=4`** cho official shape
#14. Full strict accuracy PASS `0/3,276,800,000`; final optimized-only median
`6780.3867 ms`, p90 `6792.4046 ms`, throughput `471,949.48 token/s`, peak
allocated `25.676 GiB`.

Không đổi `main.py`. V19 CUDA arithmetic bị reject cho promotion: dù correctness
PASS, full #1–#13 geomean speedup chỉ `10.3079x`, thấp hơn V16.1 cùng host
`11.8030x`; optimized latency geomean regress `13.73%`.

## Environment

- Vast.ai NVIDIA GeForce RTX 5090, SM120, 32,607 MiB.
- Driver `595.71.05`.
- Python `3.12.14`; PyTorch `2.11.0+cu128`; CUDA wheel `12.8`.
- cuDNN `9.19.0`; Triton `3.6.0`.
- Public dtype FP32, TF32 bật, seed `1234`.
- #1–#13: optimized `torch.compile(mode="max-autotune")`.
- #14 multi-stream: inner executor `max-autotune-no-cudagraphs`.

## CUDA build fix

NVCC 12.8 ban đầu reject `wmma::fill_fragment()` vì FP16 accumulator nhận
literal `float`. Source đã đổi literal zero thành `__float2half(0.0f)`. CUDA
extension sau đó build và execute thật; không bật fallback.

## V19 checkpoint sweep

Mọi mode `16/32/64/128/fp32` strict PASS năm trials trên official canaries
#7/#10/#2. Shape #6 dùng accuracy 3 trials, warmup 10, repeats `30 x 3`,
optimized max-autotune:

| Mode | Shape #6 max abs | Optimized median |
|---|---:|---:|
| K16 | `0.00183275` | `29.7888 ms` |
| K32 | `0.00167364` | `29.8112 ms` |
| K64 | `0.00180274` | `29.7675 ms` |
| K128 | `0.00207281` | `29.8196 ms` |
| FP32 control | `0.00160612` | `29.5313 ms` |
| V16.1 start/end control | `0.00160612` | `25.1593 / 25.2380 ms` |

K64 là FP16-checkpoint mode nhanh nhất, nhưng vẫn regress khoảng `18.13%` so
với trung bình V16.1 controls. FP32 control nhanh hơn mọi FP16 mode nhưng vẫn
regress khoảng `17.19%`.

Full V19 K64 #1–#13 strict PASS toàn bộ; worst max abs `0.00181192`. Geomean
speedup là `10.3079x`, còn V16.1 start control cùng host là `11.8030x`.

## Partition sweep trên shape #14

Mọi P2/P4/P8 canary đã chạy multi-stream path thật và strict PASS. P16 không
chạy: P8 đã regress và resident memory trong full-output timing lên khoảng
`29.6 GiB`, chỉ còn khoảng `2.2 GiB` so với capacity hiển thị.

V19.1.0 forward sweep, warmup 1/repeats 3:

| Parts | Median | Peak allocated |
|---:|---:|---:|
| 1 | `6877.4375 ms` | `25.649 GiB` |
| 2 | `6801.2031 ms` | `25.658 GiB` |
| 4 | `6810.6011 ms` | `25.676 GiB` |
| 8 | `6848.9131 ms` | `25.711 GiB` |
| 1 end control | `6952.9688 ms` | `25.649 GiB` |

Reverse finalists, warmup 1/repeats 5:

| Parts | Median |
|---:|---:|
| 1 start control | `6884.9097 ms` |
| 4 | `6810.3179 ms` |
| 2 | `6840.6865 ms` |
| 1 end control | `6966.1270 ms` |

P2/P4 đổi thứ tự thắng giữa hai runs; trung bình hai orders là P2
`6820.9448 ms`, P4 `6810.4595 ms`. Chênh lệch chỉ `0.15%`, nhưng P4 là số
nhanh nhất đo được và ổn định hơn giữa hai orders. Post-accuracy final P4 đạt
`6780.3867 ms`.

V19.1.1 K64 forward sweep chọn P2: P1 start/end trung bình `7257.9707 ms`, P2
`7134.1021 ms`, P4 `7144.6230 ms`, P8 `7182.6807 ms`.

## Full strict shape-#14 gates

Validator chạy candidate theo group bằng đúng số parts rồi streaming-compare đủ
32 samples; cách này exercise outer scheduler nhưng không giữ full B32 output
trong khi dựng reference. Native full B32 output contract và timing chạy riêng.

| Candidate | Configuration | Max abs | Mean abs | Failed | Accuracy peak |
|---|---|---:|---:|---:|---:|
| V19 | K64, P1 | `0.000997305` | `0.000110411` | `0/3,276,800,000` | `19.967 GiB` |
| V19.1.0 | P2 | `0.000944138` | `0.0000656367` | `0/3,276,800,000` | `20.367 GiB` |
| V19.1.0 | P4 | `0.000944138` | `0.0000656367` | `0/3,276,800,000` | `21.147 GiB` |
| V19.1.1 | K64, P2 | `0.000997305` | `0.000110411` | `0/3,276,800,000` | `20.367 GiB` |

## Final cross-candidate timing

Warmup 1, repeats 5, forward/reverse order, no CUDA Graph:

| Candidate | Tuned config | Forward median | Reverse median | Two-order average |
|---|---|---:|---:|---:|
| V19 | K64, P1 | `7251.4170 ms` | `7310.5811 ms` | `7280.9991 ms` |
| V19.1.0 | P2 control | `6836.2656 ms` | `6866.1680 ms` | `6851.2168 ms` |
| V19.1.1 | K64, P2 | `7173.3130 ms` | `7185.5513 ms` | `7179.4322 ms` |

Accuracy-valid winner V19.1.0 P4 post-gate median is `6780.3867 ms`. So với
two-order V19 average, winner nhanh hơn khoảng `6.88%`; so với V19.1.1 P2,
nhanh hơn khoảng `5.56%`. Trong hai P1-control sandwiches riêng, P4 cải thiện
V16.1 parent khoảng `1.51–1.66%`.

Shape #14 vẫn chỉ có optimized-only latency: original baseline cần attention
score khoảng `18.6 TiB`, nên không báo paired official speedup.
