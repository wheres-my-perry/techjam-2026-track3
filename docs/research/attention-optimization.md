# Research report: tối ưu Attention và Transformer trên GPU

> Ngày chốt nguồn: 2026-08-30
>
> Mục tiêu: tìm các implementation và ý tưởng có code công khai để tối ưu Transformer layer của TikTok TechJam 2026 Track 3 trên RTX 5090, đồng thời phân biệt rõ phương án dùng được ngay, phương án cần port, và phương án thay đổi ngữ nghĩa.
> Trạng thái số liệu: mọi con số từ bên ngoài đều được ghi là **claim của tác giả**. Phần follow-up §18.3 tách riêng số đã đo bởi repository trên exact shape #14; isolated/inner diagnostics ở đó vẫn không phải paired official full-model speedup.

## 1. Kết luận ngắn

Report liệt kê **137 entry implementation/pattern** trong 8 catalogue; một số thư viện xuất hiện ở nhiều catalogue vì có cả attention, GEMM, quantization và runtime components.

Sau follow-up đã đo trên RTX 5090, trạng thái và thứ tự ưu tiên hiện tại là:

1. **Giữ PyTorch Flash + whole-model batch microchunk làm control đã ship.**
   Path này đã PASS full #1–#14 và làm shape #14 executable trên 32 GiB.
2. **Probe exact FlashInfer SM120 trước.** Đây là exact vendor/library backend
   quan trọng duy nhất trong shortlist chưa chạy trên exact #13/#14 workload;
   mọi adapter và layout copy phải nằm trong timing.
3. **Fuse LayerNorm + QKV projection + backend-native layout**, đồng thời thử
   separate/no-concat QKV activation cho #14. Nguồn chính là Transformer Engine
   `LayerNormLinear`, CUTLASS epilogue permutation và FlashInfer separate Q/K/V.
4. **Xây accuracy-aware workload router.** Direct-layout QKV đã có measured
   upside trên #6/#13, nhưng route mới phải dùng predicate theo workload/GPU,
   không hard-code official test ID.
5. **Mở rộng robustness và portability** qua nhiều seed/input scale/padding,
   backend logging, compile cold-start và GPU/software stack thứ hai.
6. **Để exact FFN/small-shape kernel sau bottleneck chính.** cuBLASLt/TE/CUTLASS
   và `Dh=8`/`D=32` specialization chỉ bắt đầu sau profile đúng target shape.
7. **Giữ low precision ở nhánh high-risk/deferred.** Current SageAttention có
   isolated upside nhưng fail strict model gate; chỉ recipe mới có protection
   cho outlier/precision boundary mới đáng accuracy-only probe lại.
8. **Chỉ viết custom SM120 attention khi library path chạm trần.** Khi đó mới
   cân nhắc online softmax, causal triangular load balancing, Triton, CUTLASS,
   TileLang hoặc ThunderKittens. `sm100` datacenter Blackwell và `sm120` GeForce
   Blackwell không phải cùng một target kernel.

PyTorch Flash đã thắng cuDNN, Efficient và FA4 b28 ở exact shape-#14
attention/inner workload. Sage PV-FP16 nhanh isolated nhưng fail strict
full-model; exact-prefix correction tiếp tục fail official #6/#9. Chi tiết và
ranh giới claim nằm ở §18.3, `EXPERIMENTS.md` §17–18 và D-034/D-040.

Các hướng không nên ưu tiên cho điểm thi hiện tại:

- FlashAttention-3: Hopper-only, không phải RTX 5090.
- FP8 DPA của Transformer Engine: selector hiện tắt path này trên `sm120`.
- FlashMLA, XQA, PagedAttention, LeanAttention: rất hay nhưng chủ yếu cho MLA hoặc decode/KV cache, không khớp full self-attention prefill của benchmark.
- Longformer, BigBird, Performer, Linformer, Reformer, MoBA, NSA: thay đổi attention operator hoặc cần retrain; gần như chắc chắn không drop-in qua strict elementwise comparator.
- CPU kernels: hữu ích cho portability hoặc host overhead, nhưng timed forward chính chạy CUDA; CPU không thay GPU compute được.

## 2. Ràng buộc dùng để đánh giá

### 2.1 Accuracy là cổng bắt buộc

Mỗi phần tử phải thỏa:

```text
absolute_error < 0.002 OR relative_error < 0.02
```

Đây là strict `<`, không phải `<=`. Vì vậy report dùng các nhãn:

| Nhãn | Ý nghĩa |
|---|---|
| **E0** | Exact cùng attention semantics; chỉ thay lịch thực thi, tiling, fusion hoặc layout. Vẫn phải đo sai số do thứ tự cộng/dtype. |
| **E1** | Exact về công thức nhưng cần port/custom kernel; rủi ro numerical hoặc integration cao hơn. |
| **Q** | Quantized/approximate; có khả năng qua comparator nhưng không được giả định. Accuracy-only trước, benchmark sau. |
| **A** | Thay đổi attention graph/sparsity/low-rank; thường phải retrain hoặc đổi weight. Không drop-in cho bài thi. |
| **D** | Decode/KV-cache/distributed/CPU-specific; đáng học ý tưởng nhưng không khớp workload chính. |

### 2.2 RTX 5090 là `sm120`, không phải `sm100`

[NVIDIA công bố RTX 5090 có 32 GB GDDR7 và 1.792 TB/s peak memory bandwidth](https://www.nvidia.com/en-gb/geforce/graphics-cards/50-series/). Các kernel “Blackwell” cho B100/B200 thường target `sm100`; RTX 50 target `sm120`. Một số khác biệt có tác động trực tiếp:

- CUTLASS có riêng [example 79 cho Blackwell GeForce SM120](https://github.com/NVIDIA/cutlass/tree/main/examples/79_blackwell_geforce_gemm).
- Code FA4 cho RTX 50 có file riêng [`flash_fwd_sm120.py`](https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/cute/flash_fwd_sm120.py), nhưng hiện dùng lại SM80-era MMA path và giới hạn shared memory 99 KB; đã từng có [issue compile/epilogue riêng trên SM120](https://github.com/Dao-AILab/flash-attention/issues/2453), nên cần pin và xác minh version thực tế.
- [Transformer Engine](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/attention/attention.html) xếp backend `sm100/sm120` theo `cuDNN attention > flash-attention > PyTorch-native`, và FP8 DPA hiện chỉ eligible trên `sm100`, không phải `sm120`.
- FlashMLA chính chủ hiện công bố `sm90/sm100`, không có `sm120` trong support matrix.

Do đó mọi claim “Blackwell optimized” phải kiểm tra compute capability cụ thể trước khi đầu tư port.

## 3. Bản đồ 14 test shape

| Nhóm | Test | Đặc trưng | Bottleneck có khả năng cao | Hướng nên thử trước |
|---|---:|---|---|---|
| Batch sweep | #1–#6 | `S=128,D=128,H=4,L=4`, `B=1…10000` | B nhỏ: launch/dispatch; B lớn: GEMM, memory traffic, FFN | CUDA Graph, full-block compile, fused LN/QKV, fused FFN, cublasLt/CUTLASS GEMM |
| Hidden sweep | #7–#8 | `D=32` và `D=1024` | D=32: tiny GEMM/launch; D=1024: tensor-core GEMM/FFN | Persistent/tiny kernels cho #7; cublasLt/CUTLASS/TE cho #8 |
| Head sweep | #9–#11 | Head dim `128,64,8` | Tile shape, occupancy, vectorization thay đổi mạnh | Route theo `head_dim`, không dùng chung một attention config |
| Sequence sweep | #12–#13 | `S=32` và `S=1024` | S=32: overhead; S=1024: attention IO/compute | Full-block fusion/CUDA Graph cho #12; cuDNN/FA4/FlashInfer/Sage cho #13 |
| Extreme | #14 | `B=32,S=100000,D=1024,H=16,L=2` | Peak memory và exact causal attention O(S²) compute | Whole-model batch microchunk + online softmax + preallocated output; backend exact SM120 |

### 3.1 Các specialization cụ thể

- **#2, B=1**: kernel launch và Python/dispatcher overhead có thể lớn hơn compute. `torch.compile(mode="reduce-overhead")`, fullgraph và CUDA Graph đáng thử hơn một GEMM kernel phức tạp.
- **#6, B=10000**: số token phẳng là 1.28 triệu. Attention mỗi sequence vẫn ngắn, còn QKV/out/FFN trở thành các GEMM rất lớn. Đây là shape tốt nhất để thử cublasLt heuristic, CUTLASS persistent GEMM, FROST nếu support SM120 được xác nhận, và FFN fusion.
- **#7, D=32, head_dim=8**: Tensor Core padding và library setup có thể ăn hết lợi ích. Persistent small-GEMM hoặc một transformer-block kernel chuyên dụng có thể thắng.
- **#8, D=1024, head_dim=256**: QKV/FFN tốn nhiều hơn. Đo attention-only sẽ dẫn tới quyết định sai; phải profile whole layer.
- **#11, head_dim=8**: cần tile riêng. Một config tối ưu cho `Dh=64/128` thường lãng phí register/shared-memory ở đây.
- **#12, S=32**: ghép LayerNorm → QKV → attention → out/residual và LayerNorm → FFN có giá trị lớn vì từng kernel rất ngắn.
- **#13, S=1024**: là shape “thật” tốt nhất để phân biệt SDPA backend, tiling và SageAttention mà chưa có áp lực memory cực đoan của #14.

## 4. Shape #14: phân tích memory và compute

### 4.1 Các con số bắt buộc phải biết

Với `B=32,S=100000,D=1024,H=16`:

| Tensor/khối | Kích thước |
|---|---:|
| Input FP32 `[B,S,D]` | 12.207 GiB |
| Output FP32 `[B,S,D]` | 12.207 GiB |
| `input + reference + candidate` FP32 | 36.621 GiB |
| Packed QKV FP16 toàn batch | 18.311 GiB |
| Dense attention score FP32 `[B,H,S,S]` | 18.626 TiB |
| Causal QK+PV, một layer | khoảng 655.36 TFLOP |
| Causal QK+PV, hai layer | khoảng 1.31072 PFLOP |

Hai kết luận:

1. Baseline materialize score dense không thể chạy trên RTX 5090. Flash/online softmax không phải optimization tùy chọn mà là điều kiện tồn tại.
2. Accuracy harness hiện tại giữ `x`, baseline output và optimized output cùng lúc; riêng ba tensor đó đã vượt 32 GB. Đây là blocker của **cách validate local**, không phải lý do nới comparator.

### 4.2 Phương án exact nên dùng

**Whole-model batch microchunk**:

1. Preallocate final output FP32 `[32,100000,1024]`.
2. Lấy slice batch `x[b0:b1]`.
3. Chạy slice đó xuyên cả hai Transformer layers.
4. Ghi kết quả vào slice tương ứng của output.
5. Tái sử dụng cùng scratch buffers cho chunk tiếp theo.

Sample trong batch không tương tác, nên phép biến đổi này giữ nguyên semantics. Khác biệt numerical chỉ đến từ kernel/dtype, không từ microbatching.

**Vì sao phải chạy xuyên toàn bộ layer theo chunk**: nếu làm layer 1 cho tất cả chunk rồi layer 2, phải giữ activation toàn batch giữa hai layer, mất phần lớn lợi ích memory. Chạy một chunk xuyên toàn model chỉ cần input gốc, output đích, activation của chunk và workspace.

**Chunk size** nên autotune từ nhỏ lên lớn (`1,2,4,8`) theo peak memory và latency. Không hard-code theo dung lượng danh nghĩa; cuDNN/Flash/CUDA Graph có workspace và allocator cache riêng.

### 4.3 Validator streaming cho local accuracy

Để kiểm tra đúng luật mà không giữ ba output full-size:

- Chạy baseline/reference theo một batch slice có thể fit.
- Chạy optimized cùng slice.
- Tính `abs_error`, `relative_error`, boolean pass và các cực trị ngay trên slice.
- Chỉ giữ scalar aggregate cùng vị trí phần tử xấu nhất; giải phóng hai output slice trước chunk tiếp.
- Không đổi `rtol`, `atol`, input, seed hoặc công thức OR.

Đây chỉ là thay cách đo local để tránh OOM; performance benchmark vẫn phải dùng forward đầy đủ và shape output đầy đủ.

### 4.4 Nếu batch microchunk vẫn chưa đủ

Thứ tự fallback exact:

1. Giảm batch chunk tới 1.
2. Tránh packed QKV full chunk nếu backend nhận Q/K/V riêng; TensorRT-LLM và FlashInfer đều có bài học “no concat”.
3. Fuse QKV projection với layout/store để không có tensor transpose/copy phụ.
4. Query-chunk streaming với K/V resident và online softmax. Đây là custom E1, phức tạp hơn nhiều vì QK/PV vẫn phải quét toàn causal K/V.
5. CPU offload chỉ cho validator/reference hoặc khi luật/harness chính thức cho phép; không nên đưa PCIe transfer vào timed optimized forward.

## 5. Shortlist hành động sau follow-up RTX 5090

Các nhãn dưới đây phản ánh **trạng thái hiện tại**, không phải mức hấp dẫn lý
thuyết lúc bắt đầu research:

- `SHIPPED`: thuộc standalone final và đã qua official gate.
- `NEXT`: candidate chưa đo quan trọng nhất, có experiment gate cụ thể.
- `DEFERRED`: chỉ mở lại sau khi candidate ưu tiên hơn hoặc blocker hiện tại được
  giải quyết.
- `REJECTED`: đã có evidence chặn promotion theo correctness hoặc end-to-end
  performance; không rerun nếu không có thay đổi bản chất.

| Status | Candidate | Evidence / reason | Next gate |
|---|---|---|---|
| `SHIPPED` | PyTorch Flash + whole-model batch microchunk | Full #1–#14 strict PASS; #14 native B32 chạy trong 32 GiB | Giữ làm control cho mọi candidate |
| `NEXT` | FlashInfer `fmha_v2_prefill_sm120` | Exact SM120 backend quan trọng duy nhất trong shortlist chưa probe | #13 → #14 B1; tính toàn bộ adapter/layout cost trong timing |
| `NEXT` | `LayerNorm → QKV → backend-native layout` | Giảm launch/read-write và có thể bỏ `permute/contiguous`; đặc biệt hữu ích khi nối với FlashInfer | Gate state dict, epsilon/math order, rồi whole-layer #6/#8/#13/#14 B1 |
| `NEXT` | Separate/no-concat QKV activation cho #14 | Packed weight vẫn hữu ích nhưng packed activation full batch tạo áp lực 18.311 GiB | Đo peak memory và latency cùng exact attention backend |
| `NEXT` | Accuracy-aware workload router | Direct-layout QKV đã thắng #6 khoảng `3.43%` và #13 `0.98–2.20%`, nhưng force-all không thắng | Predicate theo workload/GPU, full matrix, reverse order, aggregate gate |
| `NEXT` | Robustness và portability gate | Final mới khóa một seed và một RTX 5090 software stack | Multi-seed/scale/padding, backend log, cold start, GPU/software thứ hai |
| `DEFERRED` | cuBLASLt/TE/CUTLASS exact FFN và kernel `Dh=8`/`D=32` | Có fit cho #6/#8 hoặc #7/#11 nhưng không phải bottleneck #14 | Chỉ bắt đầu sau profile mới trên target shape |
| `DEFERRED` | Protected SageAttention2++ precision island | Current Sage recipe có isolated upside nhưng fail strict full model | Recipe mới phải PASS accuracy-only #1/#8/#13 trước timing |
| `DEFERRED` | Larger chunks hoặc multi-stream #14 | B=2 chỉ nhanh hơn `0.30–0.59%`; attention vẫn chiếm `92.258%` | Chỉ mở lại khi profiler chứng minh launch gap hoặc memory headroom mới |
| `REJECTED` | cuDNN, Efficient và FlashAttention-4 cho exact #14 inner path | cuDNN chậm `2.38–2.92%`, Efficient gần `2x`, FA4 chậm `7.72%` so PyTorch Flash | Chỉ reconsider trên hardware/software khác hoặc implementation đổi bản chất |
| `REJECTED` | Current Sage automatic/corrected recipes | Automatic quantization fail mạnh; corrected full matrix vẫn fail #6/#9 | Không có performance claim hoặc promotion |
| `REJECTED` | Sparse/linear/low-rank submission path | Thay dense attention semantics hoặc cần retrain | Chỉ giữ làm research ngoài submission |

## 6. Catalogue A — exact attention kernels và runtime

| Implementation | Code/nguồn | Điểm nổi bật | Memory/precision | Đánh giá cho repo |
|---|---|---|---|---|
| PyTorch SDPA | [API](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html) | Một API dispatch Flash, memory-efficient hoặc math backend; dễ ép backend để benchmark công bằng | Flash path không materialize S² | **E0, đã dùng**; cần log backend thật thay vì chỉ nhìn API |
| cuDNN FusedAttention | [cuDNN Attention](https://docs.nvidia.com/deeplearning/cudnn/v1.13.0/operations/Attention.html) | SDPA flash-based, causal/padding/ragged/GQA, vendor-tuned | FP16/BF16; có FP8 path trên một số GPU | **E0, measured control**; exact #14 inner chậm `2.38–2.92%` so PyTorch Flash |
| Transformer Engine DotProductAttention | [Docs](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/attention/attention.html) | Backend selector và debug flags; ưu tiên cuDNN trên Blackwell | Có layout `bshd/sbhd/thd`, precision policy | **E0**; dùng như harness so backend, không nhất thiết thay toàn model |
| FlashAttention 1 | [Code/paper](https://github.com/Dao-AILab/flash-attention) | IO-aware tiled exact attention, online softmax | O(S) activation memory thay O(S²) | Nền tảng thuật toán; FA2/4 phù hợp hơn |
| FlashAttention 2 | [Code/paper](https://github.com/Dao-AILab/flash-attention) | Giảm non-matmul FLOP, partition tốt hơn theo sequence/head | FP16/BF16, head dim tới 256 | **E0**; current SDPA Flash có thể tương đương |
| FlashAttention 3 | [Paper/code](https://github.com/Dao-AILab/flash-attention/tree/main/hopper) | Warp specialization, TMA, overlap Tensor Core/softmax, FP8 incoherent processing | Hopper H100/H800 | **Không target SM120**; chỉ học scheduling |
| FlashAttention 4 CuTeDSL | [Code](https://github.com/Dao-AILab/flash-attention/tree/main/flash_attn/cute) | Một codebase CuTeDSL cho SM80/90/100/120, causal/local/block sparse | FP16/BF16 trên SM120; FP8 FA4 chỉ SM100 theo interface hiện tại | **E0/E1, measured reject**; exact #14 attention chậm `7.72%` so PyTorch Flash |
| FlashInfer FMHA v2 | [Attention API](https://docs.flashinfer.ai/api/attention.html) | Có `fmha_v2_prefill_sm120`, separate contiguous Q/K/V, task-scheduled primitives | Prefill exact; nhiều layout/KV wrapper | **E0/E1, NEXT** cho #13/#14 |
| xFormers memory-efficient attention | [Code](https://github.com/facebookresearch/xformers) | Exact MEA với CUTLASS/Flash backends, attention bias abstractions | O(S) extra memory | **E0, DEFERRED** cross-check; có thể chỉ dispatch lại FA/CUTLASS |
| Triton fused attention tutorial | [Kernel](https://github.com/triton-lang/triton/blob/main/python/tutorials/06-fused-attention.py) | FA2-style kernel dễ sửa BLOCK_M/N, stages, warp specialization | FP16, FP8 sample, FP32 softmax state | **E1, DEFERRED**; chỉ custom khi library path chạm trần |
| CUTLASS FMHA/Grouped GEMM MHA | [CUTLASS](https://github.com/NVIDIA/cutlass) | CuTe layouts, grouped GEMM MHA, epilogue visitors | Vendor primitives, nhiều dtype | **E1**; code mine tốt, integration cao |
| TensorRT-LLM Context FMHA | [Attention docs](https://nvidia.github.io/TensorRT-LLM/1.2.0/features/attention.html) | Fused context MHA; packed tokens; separate backend for context/decode | Flash algorithm, no S² score | **E1/D**; adapter engine nặng, nhưng ý tưởng no-padding/no-concat rất hữu ích |
| ONNX Runtime fused attention | [CUDA kernel map](https://github.com/microsoft/onnxruntime/tree/main/onnxruntime/contrib_ops/cuda/bert) | Fused MHA, CUTLASS memory-efficient and Flash kernels, graph rewrites | FP16 Flash; FP32 memory-efficient path | **E1**; code mine/cross-check, không ưu tiên engine conversion |
| FasterTransformer BERT | [Code](https://github.com/NVIDIA/FasterTransformer), [guide](https://github.com/NVIDIA/FasterTransformer/blob/main/docs/bert_guide.md) | Fused MHA, remove padding, GEMM config search, fused residual/LN/FFN patterns | FP32/FP16/BF16/INT8; code cũ | **E1**; đào pattern, không port nguyên framework |
| LightSeq | [Code](https://github.com/bytedance/lightseq) | CUDA/cuBLAS/CUB Transformer inference, fused layer kernels | FP16/INT8 | **E1**; code cũ nhưng nhiều fusion thực chiến |
| ByteTransformer | [Code](https://github.com/bytedance/ByteTransformer) | Padding-free BERT, variable-length packing, fused attention | Giảm compute/memory khi có padding | **E1**; chỉ đáng giá khi valid-token mask có padding thực |
| DeepSpeed Inference kernels | [Code](https://github.com/deepspeedai/DeepSpeed), [tutorial](https://github.com/deepspeedai/DeepSpeed/blob/master/docs/_tutorials/inference-tutorial.md) | Kernel injection, custom attention/MLP, CUDA và Triton | FP16/INT8 | **E1**; mine `GELUGemmOp` và fused block patterns |
| AITemplate | [Code](https://github.com/facebookincubator/AITemplate) | Compile graph thành CUDA/HIP, vertical/horizontal/memory fusion | Chủ yếu FP16 inference | **E1**; static 14 shapes hợp AOT, nhưng port model tốn công |
| TileLang FlashAttention examples | [Code](https://github.com/tile-ai/tilelang) | TVM/TIR DSL, TMA/WGMMA-style pipelining, autotune-friendly | FP16 với FP32 accum trong ví dụ | **E1**; kiểm tra SM120 thật trước |
| ThunderKittens | [Code](https://github.com/HazyResearch/ThunderKittens) | Header-only tile primitives, Blackwell support, worker overlap, attention kernels | FP16/BF16/low precision tùy kernel | **E1, ý tưởng độc**; tốt cho full custom path |
| Rabe–Staats memory-efficient exact attention | [Paper](https://arxiv.org/abs/2112.05682) | Chunk/recompute exact attention với memory rất thấp | O(1)/O(log n) lý thuyết, practical chunking | **E1**; concept fallback cho #14, Flash kernels production-ready hơn |

## 7. Catalogue B — QKV, LayerNorm, layout và padding

### 7.1 Các implementation/pattern đáng đào

| Implementation/pattern | Code/nguồn | Điểm nổi bật | Đánh giá cho repo |
|---|---|---|---|
| Single packed QKV GEMM | [FlashAttention MHA](https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/modules/mha.py) | Ghép `Wq,Wk,Wv` thành `[D,3D]`, một GEMM thay ba GEMM/launch | **E0, đã có từ V1**; vẫn cần tối ưu output layout |
| FlashAttention `FusedDense` | [Source](https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/ops/fused_dense.py) | Fused matmul+bias; implementation tham khảo cho QKV/out projection | **E1**; code chủ yếu train/A100-era, đo lại trên SM120 |
| Transformer Engine `LayerNormLinear` | [API](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/api/pytorch.html) | LayerNorm rồi Linear trong module fused; có thể trả LN output cho residual | **E0/E1, NEXT**; map epsilon, affine params và dtype chính xác |
| Transformer Engine fused QKV params | [Transformer source](https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/pytorch/transformer.py) | `fuse_qkv_params`, QKV layout/interleaving và TE attention tích hợp | **E1**; tham khảo cách tổ chức weight/state_dict |
| CUTLASS GEMM epilogue permutation | [Changelog](https://github.com/NVIDIA/cutlass/blob/main/CHANGELOG.md) | GEMM store thẳng vào layout attention cần, bỏ transpose/contiguous kernel | **E1, NEXT**; rất hợp packed QKV → `[B,S,3,H,Dh]` |
| AITemplate memory fusion | [Code](https://github.com/facebookincubator/AITemplate) | Fuse GEMM với concat/split/slice/layout operation | **E1**; cùng mục tiêu với no-copy QKV |
| TensorRT-LLM separate Q/K/V input | [Optimization write-up](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/blogs/tech_blog/blog14_Scaling_Expert_Parallelism_in_TensorRT-LLM_part3.md) | Tránh concat Q/K/V khi kernel nhận riêng ba tensor | **E0/E1**; đặc biệt đáng giá #14 khi packed QKV tăng peak memory |
| FlashInfer separate SM120 Q/K/V | [API](https://docs.flashinfer.ai/api/attention.html) | `fmha_v2_prefill_sm120(query,key,value,...)` nhận Q/K/V contiguous riêng | **E0/E1, NEXT** cho nhánh no-concat |
| FlashAttention packed API | [README](https://github.com/Dao-AILab/flash-attention) | `flash_attn_qkvpacked_func` tránh một số concat/copy ở training | **E0**; inference cần đo packed so với separate API |
| Remove padding / packed tokens | [FasterTransformer BERT](https://github.com/NVIDIA/FasterTransformer/blob/main/docs/bert_guide.md) | Gom valid token, attention dùng offsets, scatter output trở lại | **E0/E1** nếu mask có padding; overhead có thể thua khi tất cả token valid |
| ByteTransformer padding-free | [Code](https://github.com/bytedance/ByteTransformer) | Tối ưu toàn encoder quanh variable-length packed input | **E1**; source tốt cho mask path |
| TensorRT-LLM packed tensors | [Docs](https://nvidia.github.io/TensorRT-LLM/1.2.0/features/attention.html) | Không pad batch đến max length; giảm surrounding GEMM lẫn attention | **E1/D**; áp dụng ý tưởng, không cần port engine |
| Prepacked inference weights | [Current Flash/FT/TE patterns](https://github.com/NVIDIA/TransformerEngine) | Cast/transpose/interleave weight một lần sau `load_state_dict()` | **E0**; cache phải refresh khi state_dict đổi và không dùng sai trong training |
| Separate QKV multi-stream | [CUDA streams concept](https://docs.nvidia.com/cuda/cuda-c-programming-guide/) | Ba projection độc lập có thể overlap khi GEMM quá nhỏ để saturate GPU | **E1**; chỉ thử #2/#7 sau packed GEMM, thường packed vẫn tốt hơn |
| QKV GEMM shape padding | [torch.compile options](https://docs.pytorch.org/docs/stable/generated/torch.compile) | Inductor `shape_padding` align GEMM cho Tensor Core | **E0** về semantics; profile vì D=32/Dh=8 có thể tốn compute thừa |

### 7.2 Điểm thiết kế quan trọng

- Fuse QKV không đồng nghĩa phải materialize một packed QKV khổng lồ. Với #14, có thể dùng một GEMM output có ba view/offset hoặc kernel attention nhận ba pointer riêng.
- Layout conversion cần được tính trong latency. Một attention kernel nhanh hơn 10% nhưng đòi `permute().contiguous()` có thể thua end-to-end.
- LayerNorm cần giữ precision island. Repository hiện cho thấy FP32 ngay trước GELU quan trọng; tương tự, không nên tự động hạ LayerNorm/QKV accumulator nếu chưa qua matrix accuracy.
- `valid_token_mask` có hai chi phí: compact/scatter và nhánh causal+padding trong attention. Nên route packed-token chỉ khi số invalid token đủ lớn.

## 8. Catalogue C — FFN, GELU, GEMM và epilogue fusion

| Implementation | Code/nguồn | Điểm nổi bật | Precision/caveat | Đánh giá |
|---|---|---|---|---|
| cuBLASLt epilogue | [Docs](https://docs.nvidia.com/cuda/cublas/index.html) | `BIAS`, `GELU`, `GELU_BIAS`, aux output trong GEMM epilogue | GELU/compute order phải so strict comparator | **E1, DEFERRED** exact FFN candidate cho #6/#8 |
| FlashAttention fused dense/MLP | [Source](https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/ops/fused_dense.py) | Matmul+bias+GELU và hai-layer MLP wrapper | `FusedMLP` dùng `gelu_approx`; source ghi Hopper fused path có thể chậm hơn unfused | **E1/Q**; học heuristic, không copy mù |
| Transformer Engine `LayerNormMLP` | [API](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/api/pytorch.html) | Fuses norm + MLP module, giảm launch/memory traffic | TE thường thiên low precision/GLU; kiểm tra exact GELU | **E0/E1, DEFERRED** |
| Transformer Engine op fuser | [Docs](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/op_fuser/op_fuser.html) | Compose LayerNorm/Linear/activation/Linear rồi fuser nhận pattern | Có thể thử chỉ inference, static shapes | **E1** |
| CUTLASS GEMM→LayerNorm→GEMM | [Example 37](https://github.com/NVIDIA/cutlass/tree/main/examples/37_gemm_layernorm_gemm_fusion) | Split/fuse LayerNorm vào GEMM trước và sau; Shift-K cho variance numerical | Example cũ target Ampere; concept port lên SM120 | **E1, ý tưởng mạnh** |
| CUTLASS back-to-back GEMM | [CUTLASS](https://github.com/NVIDIA/cutlass) | Giữ intermediate giữa hai GEMM gần compute, giảm HBM | GELU giữa GEMM làm fusion phức tạp | **E1** |
| CUTLASS EVT/epilogue visitor | [CUTLASS examples](https://github.com/NVIDIA/cutlass/tree/main/examples) | Chain bias, activation, residual, scale trong epilogue | SM100 examples không tự động chạy SM120 | **E1** |
| CUTLASS SM120 example 79 | [Code](https://github.com/NVIDIA/cutlass/tree/main/examples/79_blackwell_geforce_gemm) | Persistent warp-specialized schedule, cluster launch control, block-scaled MMA cho RTX50 | Nhiều sample là NVFP4/MXFP8, không phải FP16 exact | **E1/Q, DEFERRED**; mine scheduler/epilogue |
| cuDNN Frontend FROST | [Code](https://github.com/NVIDIA/cudnn-frontend) | JIT Blackwell matmul/grouped matmul + chained pointwise epilogues | README nhấn B200/GB200; xác nhận SM120 trước | **E1**; thử như candidate, không giả định support |
| DeepGEMM | [Code](https://github.com/deepseek-ai/DeepGEMM) | JIT persistent GEMM, TMA/warp specialization, fine-grained scaling, fused ops | Chủ lực FP8/FP4 và SM90/SM100; SM120 không chắc | **E1/Q**; học scheduling hơn là drop-in |
| tiny-cuda-nn FullyFusedMLP | [Code](https://github.com/NVlabs/tiny-cuda-nn) | Giữ small MLP trong shared/register; JIT whole-kernel fusion | Hidden width chỉ 16/32/64/128 cho FullyFusedMLP; activation không khớp GELU mặc định | **E1/A**; rất đáng học cho #7/D=32 và D=128 |
| Triton matmul tutorial | [Code](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html) | Grouped program ordering tăng L2 reuse, autotune block sizes | Custom GEMM thường khó thắng cuBLAS ở shape lớn | **E1**; dùng cho fusion/small shape |
| Triton persistent matmul | [Code](https://triton-lang.org/main/getting-started/tutorials/09-persistent-matmul.html) | Persistent scheduling giảm launch/tile scheduling overhead | Cần tune per architecture | **E1, DEFERRED** cho #7 hoặc #6 |
| Triton fused softmax | [Code](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html) | Một kernel thay nhiều op, giữ row trong SRAM | Attention production đã fuse softmax | Hữu ích cho diagnostics/unfused fallback |
| Triton LayerNorm | [Code](https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html) | Row-wise reduction, fused affine | D=32/128/1024 cần config khác nhau | **E1**; nền cho LN+QKV custom |
| AITemplate vertical/horizontal fusion | [Code](https://github.com/facebookincubator/AITemplate) | Fuse pointwise/reduction/layout vào GEMM; fuse nhiều independent ops | AOT/static shape thuận lợi cho 14 shape | **E1** |
| FasterTransformer FFN | [Code](https://github.com/NVIDIA/FasterTransformer) | Fused activation, bias, residual, LayerNorm, GEMM algo cache | Code cũ, architecture support cũ | **E1** code mine |
| DeepSpeed GELUGemm | [Code](https://github.com/deepspeedai/DeepSpeed) | Custom CUDA/Triton MLP operators và kernel injection | Blog DeepSpeed từng thấy CUDA MLP nhanh hơn Triton | **E1** code mine |
| LightSeq fused FFN | [Code](https://github.com/bytedance/lightseq) | Full Transformer layer CUDA kernels, bias/activation/residual fusion | Phiên bản cũ | **E1** code mine |
| oneDNN Graph Gated-MLP | [Fusion patterns](https://uxlfoundation.github.io/oneDNN/v3.9/graph_fusion_patterns.html) | CPU graph nhận diện MatMul/MLP fusion | CPU path | **D**, chỉ portability |

### 8.1 Bài học cho GELU của repository

Repository đang dùng exact GELU và đã quan sát approximate GELU/low-precision ở vùng pre-GELU có thể fail. Vì vậy có ba tier:

1. **Tier an toàn**: FP32 pre-GELU, exact GELU, cast sau GELU như V11.
2. **Tier thử nghiệm exact-kernel**: GEMM FP16/TF32 nhưng accumulator/pre-activation FP32; vendor/custom epilogue exact enough.
3. **Tier high-risk**: tanh GELU, FP16 pre-GELU, FP8/INT8 FC1. Chạy accuracy matrix trước, không benchmark performance hợp lệ nếu fail.

## 9. Catalogue D — low precision và accuracy-preserving quantization

| Implementation | Code/nguồn | Điểm nổi bật | Đánh giá cho strict comparator |
|---|---|---|---|
| SageAttention 1 | [Code](https://github.com/thu-ml/SageAttention) | INT8 QK + smoothing; plug-in attention | **Q**; candidate tốt hơn naive INT8 QK |
| SageAttention 2 | [Code/paper](https://github.com/thu-ml/SageAttention) | Thorough outlier smoothing, per-thread INT4 quantization nội bộ | **Q, DEFERRED accuracy research**; current related recipes fail strict model gate |
| SageAttention 2++ | [Code](https://github.com/thu-ml/SageAttention) | Tối ưu implementation Sage2; two-level PV accumulation | **Q**; thử `QK INT8 + PV FP16` trước |
| SageAttention 3 | [Blackwell code](https://github.com/thu-ml/SageAttention/tree/main/sageattention3_blackwell) | Microscaling FP4 attention cho Blackwell | **Q high-risk**; chính tác giả khuyên Sage2 cho precision-sensitive |
| FA3 FP8 | [Paper/code](https://github.com/Dao-AILab/flash-attention/tree/main/hopper) | Block quantization + incoherent processing giảm FP8 error | **D/Q**; Hopper-only, lấy ý tưởng rotation/smoothing |
| cuDNN/TE FP8 DPA | [TE docs](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/attention/attention.html) | Cast input vào FP8, DPA FP8, output FP16/BF16; FP32 softmax stats tùy backend | **Không eligible sm120 hiện tại** |
| CUTLASS SM120 MXFP8/NVFP4 | [Example 79](https://github.com/NVIDIA/cutlass/tree/main/examples/79_blackwell_geforce_gemm) | Block-scaled Tensor Core MMA native RTX50 | **Q**; hardware hấp dẫn nhưng random weights + strict error rất khó |
| torchao | [Code](https://github.com/pytorch/ao) | Float8/int8/int4 weight/activation configs, autoquant, `torch.compile` integration | **Q**; dễ probe từng Linear, nhưng V13 đã cảnh báo FFN INT8 |
| NVIDIA Model Optimizer | [Code](https://github.com/NVIDIA/Model-Optimizer) | PTQ/QAT, FP8/NVFP4, sparsity, TensorRT export | **Q/A**; calibration/QAT ngoài phạm vi nếu weights phải giữ nguyên |
| QServe | [Code](https://github.com/mit-han-lab/qserve) | W4A8KV4 system co-design cho LLM serving | **D/Q**; decode/model-serving, không full FP32 comparator |
| Marlin | [Code](https://github.com/IST-DASLab/marlin) | FP16×INT4 weight-only GEMM, tối ưu small token batch | **D/Q**; batch/token regime và weight error không hợp benchmark |
| KIVI | [Code](https://github.com/mit-han-lab/kivi) | 2-bit KV: per-channel key, per-token value | **D/Q**; decode KV cache, không prefill self-attention |
| KVQuant | [Code](https://github.com/SqueezeAILab/KVQuant) | Outlier/attention-sink-aware KV quantization | **D/Q**; học strategy bảo vệ outlier, không drop-in |
| FBGEMM GenAI | [Code](https://github.com/pytorch/FBGEMM) | FP8/INT4 KV and GEMM kernels; current Blackwell work | **D/Q**; phần lớn serving/decode |
| FlashInfer low precision | [Code](https://github.com/flashinfer-ai/flashinfer) | FP8/FP4/NVFP4 GEMM, attention, JIT and SM120 paths | **Q/E1**; attention FP16 exact candidate trước, quant sau |
| DeepGEMM FP8/FP4 | [Code](https://github.com/deepseek-ai/DeepGEMM) | Fine-grained scale, high-throughput low precision GEMM | **Q**, target architecture caveat |
| Transformer Engine precision islands | [Docs](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/features/low_precision_training/performance_considerations/performance_considerations.html) | Fused LayerNorm quantizes trực tiếp, tránh high-precision intermediate store | **Q/E1**; ý tưởng giữ LN stats FP32 rồi hạ dtype ở biên |

### 9.1 Nhánh low-precision có xác suất cao nhất

Thứ tự nên thử:

1. Q/K INT8 có per-row/per-thread scale + smoothing; P/V FP16; output FP16; residual/output projection FP32 khi cần.
2. QK INT8 + PV FP8 với two-level accumulation.
3. Linear weight-only quantization chỉ ở projection nào profile chứng minh memory-bound.
4. FP8/NVFP4 FFN hoặc full attention sau cùng.

V13 INT8 FFN fail không chứng minh Sage QK INT8 sẽ fail: QK có smoothing, scale granularity khác và softmax có thể làm giảm hoặc khuếch đại lỗi theo phân phối. Nhưng nó đủ để cấm suy luận “INT8 chắc sẽ qua”.

## 10. Catalogue E — sparse, linear và attention thay đổi ngữ nghĩa

Nhóm này có nhiều ý tưởng sáng tạo nhất, đặc biệt cho `S=100000`, nhưng đa số không thể thay dense attention của checkpoint ngẫu nhiên rồi vẫn khớp từng phần tử. Chúng nên được dùng theo hai cách: (1) nhánh accuracy diagnostic cực nhỏ; (2) innovation demo/report, không làm implementation submit mặc định.

### 10.1 Dynamic/block sparse, training-free hoặc gần training-free

| Implementation | Code/nguồn | Điểm nổi bật | Fit |
|---|---|---|---|
| PyTorch FlexAttention | [Docs](https://docs.pytorch.org/docs/main/nn.attention.flex_attention.html) | `score_mod`, `BlockMask`, tune BLOCK_M/N, prescale QK; backend Triton/experimental Flash | **E1** nếu mask vẫn dense causal; **A** nếu thêm sparsity |
| Sparse SageAttention | [Sage repo](https://github.com/thu-ml/SageAttention) | Sage API tính nhanh bất kỳ block-sparse pattern | **Q/A**; backend tốt nếu đã có valid sparse mask |
| SpargeAttn | [Code](https://github.com/thu-ml/SpargeAttn) | Hai-stage online filter dự đoán attention map rồi softmax-aware skipping; training-free | **Q/A**; random Q/K có thể không sparse, strict error nguy hiểm |
| MInference | [Code](https://github.com/microsoft/MInference) | Nhận dạng vertical/slash/block patterns động cho long-context prefill | **A/Q**; tác giả claim tới 10x prefill A100, nhưng semantics approximate |
| FlexPrefill | [Code](https://github.com/ByteDance-Seed/FlexPrefill) | Chọn sparse pattern và budget theo context/head thời gian chạy | **A/Q**; hiện code batch=1, BF16; không khớp benchmark trực tiếp |
| MoBA | [Code](https://github.com/MoonshotAI/MoBA) | Parameter-free top-k block gating kiểu MoE | **A**; chính repo ghi cần continued training |
| FlashMoBA | [Code](https://github.com/mit-han-lab/flash-moba) | Kernel production-style cho MoBA, dựa trên FA2 | **A**; kernel hay, attention khác baseline |
| Native Sparse Attention (NSA) | [Paper](https://arxiv.org/abs/2502.11089), [kernel community](https://github.com/fla-org/native-sparse-attention) | Ba nhánh compressed/selected/sliding, hardware-aligned sparse attention | **A**; natively trainable, không drop-in |
| Flash Sparse Attention | [Code](https://github.com/Relaxed-System-Lab/Flash-Sparse-Attention) | Kernel NSA alternative, variable sparse blocks | **A/E1**; học sparse scheduling |
| DeepSeek Sparse Attention / FlashMLA | [Code](https://github.com/deepseek-ai/FlashMLA) | Token indices điều khiển sparse kernel, dense/sparse prefill cho MLA | **A/D**; MLA và sm90/sm100, không standard MHA SM120 |
| BLASST / Skip Softmax Attention | [TensorRT-LLM sparse docs](https://nvidia.github.io/TensorRT-LLM/features/sparse-attention.html) | Kernel tự skip dựa trên runtime softmax contribution | **Q/A**; ý tưởng gần Sparge, cần comparator chuyên biệt |
| RocketKV | [TensorRT-LLM docs](https://nvidia.github.io/TensorRT-LLM/features/sparse-attention.html) | Context eviction + generation top-k pages | **D/A**; KV-cache serving |
| Quest | [Paper](https://proceedings.mlr.press/v235/tang24l.html), [Code](https://github.com/mit-han-lab/Quest) | Min/max key metadata tạo upper bound, chọn top-k KV pages theo query | **D/A**; decode query-aware sparsity |
| H2O | [Code](https://github.com/FMInference/H2O) | Giữ recent + heavy-hitter KV tokens theo attention history | **D/A**; cache eviction, không full forward |
| SnapKV | [Code](https://github.com/FasterDecoding/SnapKV) | Dùng observation window để chọn KV quan trọng trước generation | **D/A** |
| DuoAttention | [Code](https://github.com/mit-han-lab/duo-attention) | Chỉ retrieval heads giữ full attention; streaming heads giữ sink+recent | **A/D**; cần học/identify head pattern |
| SparQ | [Code](https://github.com/graphcore-research/sparq-gpt-fast) | Chọn component lớn của query để approximate attention và giảm KV bandwidth | **D/Q**; decode-focused nhưng `torch.compile` implementation đáng học |
| StreamingLLM | [Code](https://github.com/mit-han-lab/streaming-llm) | Attention sinks + rolling window tạo cache hằng kích thước | **D/A** |

### 10.2 Static sparse, low-rank và linear attention

| Implementation | Code/nguồn | Điểm nổi bật | Fit |
|---|---|---|---|
| Longformer | [Code](https://github.com/allenai/longformer) | Sliding-window + global tokens, O(S·window) | **A**; cần model trained cho pattern |
| BigBird | [Code](https://github.com/google-research/bigbird) | Local + random + global sparse graph, linear complexity | **A**; repo đã archive, không lợi ở S<1024 theo README |
| Reformer | [Code](https://github.com/google/trax) | LSH attention + reversible residual | **A**; thay architecture |
| Linformer | [Code](https://github.com/facebookresearch/fairseq/tree/main/examples/linformer) | Project K/V sequence dimension xuống low rank | **A**; có learned projections, cần training |
| Performer/FAVOR+ | [Google code](https://github.com/google-research/google-research/tree/master/performer), [paper](https://arxiv.org/abs/2009.14794) | Random feature approximation cho softmax attention, linear time/memory | **A/Q**; unbiased approximation không có nghĩa elementwise pass |
| Nyströmformer | [Code](https://github.com/mlpen/Nystromformer) | Landmark/Nyström approximation của attention matrix | **A/Q** |
| Scatterbrain | [Paper](https://arxiv.org/abs/2110.15343) | Kết hợp sparse + low-rank để sửa lỗi của từng loại | **A/Q**, idea độc lạ |
| Flash Linear Attention (FLA) | [Code](https://github.com/fla-org/flash-linear-attention) | Triton/TileLang kernels cho GLA, RetNet, DeltaNet, Based, MoBA… | **A**; thư viện kernel cực tốt nhưng operator khác softmax MHA |
| Gated Linear Attention | [FLA](https://github.com/fla-org/flash-linear-attention) | Recurrent/chunk-parallel linear attention với gate | **A**, cần retrain |
| RetNet/DeltaNet/KDA/GDN | [FLA](https://github.com/fla-org/flash-linear-attention) | O(S) state update, chunkwise parallel kernels | **A**, học tiling/prefix-scan, không drop-in |

### 10.3 Vì sao sparse có xác suất thấp trên benchmark này

Weights và input benchmark là ngẫu nhiên, không phải model đã học attention locality/sparsity. Với Q/K ngẫu nhiên:

- softmax mass có thể phân tán trên nhiều key;
- top-k/block filter khó dự đoán một phần tử sẽ không quan trọng cho mọi query;
- sai số ở attention output đi qua output projection, residual, LayerNorm và hai layer, có thể khuếch đại;
- comparator là elementwise, không phải perplexity, FID hoặc end-task metric.

Do đó Sparge/Sage sparse có thể rất nhanh trên model thực nhưng vẫn fail Track 3. Giá trị chính của chúng ở đây là kỹ thuật online filtering, low-precision score, two-level accumulation và block scheduler.

## 11. Catalogue F — long-context exact, distributed và decode kernels

| Implementation | Code/nguồn | Điểm nổi bật | Fit |
|---|---|---|---|
| Ring Attention | [Code](https://github.com/Selimonder/ring-attention) | Chia sequence, quay K/V block qua device, online softmax | **D/E1**; exact và rất hợp S dài nhưng cần nhiều GPU |
| DeepSpeed Ulysses | [Code/write-up](https://github.com/microsoft/DeepSpeed/blob/master/blogs/deepspeed-ulysses/README.md) | All-to-all đổi sequence shards thành head shards; dùng được với FlashAttention | **D**; multi-GPU, degree bị ràng buộc bởi số heads |
| USP / Long Context Attention | [Code](https://github.com/feifeibear/long-context-attention) | Kết hợp Ulysses và Ring để phù hợp topology/head count | **D**; multi-GPU |
| BurstAttention | [Code](https://github.com/MayDomine/Burst-Attention) | Distributed IO-aware attention, overlap communication/compute | **D** |
| DistFlashAttn | [Code](https://github.com/RulinShao/LightSeq) | Distributed memory-efficient attention với overlap | **D**; training/distributed |
| DeepSpeed Ulysses-Offload/FPDT | [Write-up](https://github.com/deepspeedai/DeepSpeed/blob/master/blogs/ulysses-offload/README.md) | Sequence chunking, CPU/GPU memory hierarchy, pipeline scheduling | **D**, nhưng shape #14 có thể học chunk scheduler |
| LeanAttention | [Paper](https://arxiv.org/abs/2405.10480) | Stream-K style phân chia KV và reduce online-softmax states | **D/E1**; decode-focused, idea split-K hữu ích khi Q rất ngắn |
| Flash-Decoding | [PyTorch write-up](https://pytorch.org/blog/flash-decoding/) | Split KV length thành nhiều chunk để tăng parallelism, combine LSE | **D**; q-length nhỏ |
| Flash-Decoding++ | [Paper](https://arxiv.org/abs/2311.01282) | Flat GEMM/softmax scheduling, async softmax, heuristic | **D**; decode |
| FlashInfer paged/ragged attention | [Code](https://github.com/flashinfer-ai/flashinfer) | Batch prefill/decode wrapper, paged KV, CUDA Graph, JIT | **D/E1**; prefill SM120 path relevant, paged phần lớn serving |
| vLLM PagedAttention | [Code](https://github.com/vllm-project/vllm) | Virtual-memory style KV pages, continuous batching/chunked prefill | **D**; không cần KV cache trong benchmark |
| TensorRT-LLM chunked context | [Docs](https://nvidia.github.io/TensorRT-LLM/latest/index.html) | Chia prefill theo token budget, reuse activation memory | **D/E1**; concept cho #14, engine semantics khác |

Điểm có thể mang về single-GPU #14 không phải communication, mà là:

- state combine của online softmax `(max, sum-exp, weighted-value)`;
- tile scheduler cân bằng causal triangle;
- split theo query/KV khi occupancy thấp;
- scratch reuse và pipeline giữa load/compute/store;
- chunk budget dựa trên memory thật thay vì constant.

## 12. Catalogue G — CPU và host-side optimization

| Implementation | Code/nguồn | Điểm nổi bật | Khi nào hữu ích |
|---|---|---|---|
| oneDNN Graph SDPA | [Docs](https://uxlfoundation.github.io/oneDNN/dev_guide_graph_sdpa.html) | Graph pattern cho SDPA và fusion trên CPU | CPU fallback/reference, không GPU score |
| Intel Extension for PyTorch | [Code](https://github.com/intel/intel-extension-for-pytorch) | `ipex.optimize`, AMX/BF16/INT8, LLM fused ops | CPU portability hoặc baseline/reference nhỏ |
| LIBXSMM | [Code](https://github.com/libxsmm/libxsmm) | JIT small GEMM/BRGEMM/TPP, AVX512/AMX/Arm SVE/SME | Tiny D/head CPU path; source tốt cho size-specialized dispatch |
| FBGEMM | [Code](https://github.com/pytorch/FBGEMM) | JIT low-precision CPU GEMM + fused quantization | Quantized CPU inference/reference experiments |
| llama.cpp / ggml | [Code](https://github.com/ggml-org/llama.cpp) | Runtime ISA dispatch, quantized GEMM, CPU/GPU FlashAttention | Mine SIMD/packing/cache blocking; architecture mismatch |
| KleidiAI | [Code](https://github.com/ARM-software/kleidiai) | Stateless Arm microkernels, NEON/SVE/SME/SME2, explicit packing/tile variants | Arm CPU portability, không RTX score |
| XNNPACK | [Code](https://github.com/google/XNNPACK) | Highly tuned mobile/server CPU microkernels và operator fusion | CPU deployment, không contest path |
| ONNX Runtime MLAS/CPU attention | [Code](https://github.com/microsoft/onnxruntime) | CPU execution provider, transformer graph fusion | Reference/cross-platform |

CPU có một vai trò trực tiếp duy nhất trong timed CUDA forward: giảm host overhead. Cách làm phù hợp là CUDA Graph, precomputed descriptors/offsets, cached kernel selection và tránh Python loop nhỏ; không phải chuyển GEMM/attention sang CPU.

## 13. Catalogue H — compiler, scheduling, tiling và memory system

| Kỹ thuật/code | Nguồn | Điểm nổi bật | Shape phù hợp |
|---|---|---|---|
| `torch.compile(max-autotune)` | [Docs](https://docs.pytorch.org/docs/stable/generated/torch.compile) | Autotune ATen/Triton/template GEMM, epilogue fusion, CUDA Graph mặc định | #1–#13; đã cho kết quả tốt trong repo |
| `torch.compile(reduce-overhead)` | [Docs](https://docs.pytorch.org/docs/stable/generated/torch.compile) | CUDA Graph để giảm Python/launch overhead | #2,#7,#12 |
| `max-autotune-no-cudagraphs` | [Docs](https://docs.pytorch.org/docs/stable/generated/torch.compile) | Tách lợi ích kernel autotune khỏi extra graph memory | #14 hoặc debug memory |
| Inductor NVGEMM/CuTeDSL backend | [PyTorch blog](https://pytorch.org/blog/gemms-torchinductor-cutedsl-backend/) | Cho Inductor autotune thêm NVIDIA CuTeDSL GEMM candidate | GEMM-heavy #6/#8 |
| CUDA Graph manual/fullgraph | [Torch-TensorRT design](https://docs.pytorch.org/TensorRT/contributors/cuda_graphs.html) | Static buffers, replay cả forward bằng một launch graph | #2/#12; tránh #14 nếu graph cache làm OOM |
| Triton grouped program ordering | [Matmul tutorial](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html) | Tile order tăng L2 reuse | GEMM lớn #6/#8 |
| Persistent kernel | [Triton tutorial](https://triton-lang.org/main/getting-started/tutorials/09-persistent-matmul.html) | Một số program xử lý nhiều tile, giảm scheduler overhead | Small/tall-skinny GEMM |
| Warp specialization | [FA3](https://arxiv.org/abs/2407.08608), [CUTLASS](https://github.com/NVIDIA/cutlass) | Producer/consumer warps overlap load, MMA, softmax/store | Attention #13/#14, GEMM #6/#8 |
| Software pipelining/double buffering | [CUTLASS](https://github.com/NVIDIA/cutlass), [TileLang](https://github.com/tile-ai/tilelang) | Prefetch next K/V tile trong lúc compute tile hiện tại | S dài |
| Online softmax | [FlashAttention](https://arxiv.org/abs/2205.14135) | Gộp từng block với running max/sum để exact và ổn định | #13/#14 |
| Causal work balancing | [FlashAttention-2](https://tridao.me/publications/flash2/flash2.pdf) | Partition work tránh CTA nhàn ở tam giác causal | Tất cả shape causal, mạnh nhất #14 |
| Output epilogue fusion | [cuBLASLt](https://docs.nvidia.com/cuda/cublas/index.html), [CUTLASS](https://github.com/NVIDIA/cutlass) | Bias/activation/residual/cast/layout trong store cuối | QKV/FFN/out projection |
| Preallocated scratch arena | [TensorRT-LLM memory docs](https://nvidia.github.io/TensorRT-LLM/reference/memory.html) | Reuse activation memory giữa layer; tránh allocate/free | Tất cả, bắt buộc #14 |
| Packed weight cache | [Transformer Engine](https://github.com/NVIDIA/TransformerEngine) | Weight transform một lần, reuse inference | Tất cả; invalidate đúng sau load |
| Shape-specific dispatch table | Các library vendor đều dùng heuristic | Chọn backend/tile/precision theo exact tuple | Cả 14 shape |
| Memory-aware microbatching | Synthesis từ #14 và chunked context | Chọn chunk theo `cudaMemGetInfo`/measured peak | #14 |

## 14. Các tổ hợp ý tưởng mới cho repository

Phần này là **synthesis/inference từ các nguồn**, chưa phải claim đã được chứng minh.

### 14.1 Precision-island SageAttention

Kết hợp:

- SageAttention2 smoothing + INT8 chỉ cho QK;
- PV giữ FP16;
- softmax state/normalization giữ FP32;
- output projection và residual boundary dùng precision policy của V11/V12 khi cần.

Mục tiêu là lấy throughput INT8 QK nhưng không để lỗi lan qua PV, projection và residual. Đây có xác suất qua comparator cao hơn Sage3 FP4 hoặc full FP8 attention.

### 14.2 LN→QKV→attention layout fusion

Kết hợp TE `LayerNormLinear`, CUTLASS epilogue permutation và API separate Q/K/V của FlashInfer:

```text
FP32 input
  -> fused LayerNorm statistics/affine
  -> QKV GEMM with FP32 accumulator
  -> direct store FP16 into backend-native Q/K/V layout
  -> exact FMHA
```

Không lưu LayerNorm output full-size và không có `permute().contiguous()`. Với #14, đây vừa giảm latency vừa giảm peak memory.

### 14.3 Dual FFN backend theo accuracy

- Route an toàn: V11 exact pre-GELU path.
- Route nhanh: cuBLASLt `GELU_BIAS` hoặc TE fused MLP.
- Autotuner chỉ chọn route nhanh cho shape mà toàn accuracy matrix pass; không áp dụng global.

Một implementation có thể dùng exact fused path ở `D=1024` nhưng giữ V11 ở `D=32/128`, hoặc ngược lại.

### 14.4 Whole-model microbatch + CUDA Graph per chunk

Với #14, capture forward của một batch chunk cố định (ví dụ `Bc=1/2`) rồi replay cho các slice. Lợi ích là giảm loop launch overhead mà không cache graph cho tensor full batch. Điều kiện:

- static pointer/buffer hoặc copy vào static staging buffer;
- output slice copy/store không làm graph vi phạm;
- graph workspace không đẩy peak memory qua giới hạn.

### 14.5 Expert router theo shape và head dimension

Đề xuất 5 route:

| Route | Test | Backend mục tiêu |
|---|---|---|
| O — overhead-bound | #2, #12 | Fullgraph/CUDA Graph, aggressive fusion, persistent small kernel |
| M — standard | #1,#3,#4,#5,#7,#9,#10,#11 | Current V11 + best exact SDPA; specialization theo `Dh` |
| G — GEMM-heavy | #6,#8 | cublasLt/CUTLASS/TE QKV+FFN, attention backend phụ |
| A — attention-heavy | #13 | cuDNN vs FA4 vs FlashInfer vs Sage2 shoot-out |
| X — extreme memory | #14 | Whole-model microbatch + no-concat QKV + exact online softmax |

### 14.6 `head_dim=8` kernel riêng

#11 có `Dh=8`, #7 cũng có hidden/head rất nhỏ. Thay vì nhét vào tile `Dh=64`, có thể:

- vectorize nhiều heads/rows trong một CTA;
- giữ nhiều Q row trong register;
- tăng sequence parallelism thay vì K-dimension parallelism;
- fuse QKV split/reshape;
- tránh Tensor Core padding nếu SIMT/persistent path nhanh hơn end-to-end.

### 14.7 Accuracy-aware autotuner

Autotuner không chỉ tối thiểu latency. Với mỗi exact official shape và candidate config:

1. Chạy seeded accuracy cases cho causal, padding/no-padding và input scale cần thiết.
2. Loại candidate nếu có bất kỳ phần tử nào fail strict OR rule.
3. Trong tập còn lại, chọn median latency tốt nhất.
4. Lưu key gồm GPU UUID/CC, CUDA, PyTorch, dtype, `B,S,D,H,L,FFN,causal,mask`.

Điều này biến “precision trade-off” thành một phần của dispatch, thay vì quyết định global.

### 14.8 Separate-QKV memory mode chỉ cho #14

Packed QKV là tốt cho #1–#13 vì giảm GEMM/launch, nhưng full-batch packed FP16 QKV #14 đã 18.311 GiB. Một route riêng có thể:

- vẫn dùng một weight matrix packed;
- GEMM theo output tile/chunk;
- store vào ba vùng Q/K/V riêng mà attention đọc trực tiếp;
- hoặc compute Q theo query chunk, giữ K/V theo batch chunk.

“Packed weights” và “packed activation” là hai quyết định khác nhau; không nên buộc chúng đi cùng nhau.

### 14.9 End-to-end tile cost model

Chọn backend theo tổng:

```text
T_total = T_norm + T_qkv + T_layout + T_attention
        + T_out_proj + T_residual + T_ffn + T_allocator/launch
```

Không chọn theo attention kernel TOPS. SageAttention đặc biệt cần cộng quantization+smoothing; FA4/FlashInfer cần cộng layout/adapter; cuDNN cần cộng plan/workspace setup đã được warmup.

### 14.10 Batch–sequence 2D streaming cho #14

Nếu `Bc=1` vẫn không fit hoặc attention workspace quá lớn:

- dimension ngoài: batch microchunk exact;
- dimension trong: query-block streaming exact;
- K/V resident theo batch chunk;
- combine online softmax state theo K blocks.

Đây là route custom khó nhất nhưng vẫn giữ dense causal semantics. Nên dùng FlashAttention/CUTLASS/ThunderKittens code làm nền, không viết naive từ đầu.

### 14.11 Causal triangular load balancing riêng cho S=100000

Causal work của query block đầu và cuối khác nhau rất lớn. Có thể:

- schedule query tiles theo estimated K-block count;
- dynamic/persistent work queue;
- pair một heavy tile với một light tile;
- dùng cluster launch control nếu SM120 path hỗ trợ;
- tách diagonal tiles khỏi full tiles để bỏ mask branch.

Đây là nơi học FA2 và CUTLASS persistent scheduler có thể đem lại khác biệt thật.

### 14.12 Protect-outlier, not whole-tensor high precision

Thay vì giữ toàn tensor FP32:

- phát hiện row/channel có scale/outlier lớn ở Q/K hoặc pre-GELU;
- route phần thường qua FP16/INT8;
- route outlier correction qua FP32 nhỏ;
- cộng correction trước softmax hoặc activation.

Ý tưởng đến từ Sage smoothing, KVQuant outlier handling và mixed-precision GEMM. Đây là nhánh nghiên cứu high-risk nhưng sáng tạo, cần chứng minh overhead nhỏ hơn phần compute tiết kiệm.

## 15. Lộ trình thí nghiệm sau follow-up

Các prerequisite đã đóng: PyTorch Flash control, full shape-#14 batch
microchunk, streaming correctness oracle, target-environment manifest và inner
profile. cuDNN/Efficient/FA4 exact path đã bị loại theo performance; current
Sage recipes đã bị loại theo correctness. Vì vậy roadmap không chạy lại các
nhánh này như thể chưa có evidence.

### Phase 0 — `NEXT`: mở rộng gate và khóa baseline

- Thêm nhiều seed, input scale, padding ratio, causal/non-causal và mask modes.
- Ghi GPU/driver/CUDA/PyTorch/Triton, TF32, compile mode, actual SDPA backend,
  peak allocated/reserved và cold-start/steady-state riêng.
- Profile lại đúng target trước mỗi family; #14 giữ attention `92.258%` làm
  starting evidence, không suy sang #6/#8 hoặc hardware khác.

### Phase 1 — `NEXT`: exact FlashInfer SM120

1. Viết adapter version hóa cho `fmha_v2_prefill_sm120`.
2. Gate attention-only và whole-layer trên #13, sau đó #14 B1.
3. Giữ QKV/output layout tương đương và đưa mọi adapter/copy vào timed region.
4. Dừng nếu end-to-end không thắng PyTorch Flash sau khi layout đã tối ưu hợp lý.

### Phase 2 — `NEXT`: QKV/layout/memory fusion

- Thử TE `LayerNormLinear` hoặc custom/CUTLASS direct-layout epilogue.
- Nối thẳng sang backend-native Q/K/V; với #14 thử separate/no-concat activation
  và scratch reuse thay vì materialize packed activation lớn.
- Gate từng fusion riêng để định vị numerical error; không gộp QKV, attention và
  FFN thành một experiment không thể attribution.

### Phase 3 — `NEXT`: accuracy-aware workload router

- Route theo workload regime và environment key, gồm launch-bound, standard,
  GEMM-heavy, attention-heavy và extreme-memory.
- Candidate đầu là direct-layout QKV cho large `B*S`, `D=FFN=128` vì đã có
  measured evidence #6/#13; không dùng exact official test ID.
- Chỉ promote khi strict full matrix PASS, reverse-order/raw-device evidence giữ
  dấu, aggregate không regress và fallback vẫn đúng cho dtype/mask khác.

### Phase 4 — `DEFERRED`: exact FFN và small-shape specialization

- #6/#8: cuBLASLt exact GELU epilogue, TE LayerNormMLP hoặc CUTLASS SM120.
- #7/#11: `D=32`/`Dh=8` persistent kernel tránh general-purpose tile waste.
- Mỗi nhánh chỉ mở khi profiler đúng shape xác nhận bottleneck; isolated kernel
  hoặc kernel-count reduction không đủ.

### Phase 5 — `DEFERRED`: protected low precision

- Chỉ thử recipe khác bản chất: QK INT8 có smoothing, PV FP16/two-level
  accumulation, FP32 softmax/projection/residual và outlier correction nhỏ.
- Chạy accuracy-only #1/#8/#13 trước; fail thì dừng, không timing chính thức.
- Sage3 FP4/NVFP4 và blanket quantization đứng cuối vì strict comparator và
  evidence negative hiện tại.

### Phase 6 — `DEFERRED`: custom SM120 attention

Chỉ bắt đầu khi Phase 1–2 chứng minh library ceiling. Candidate gồm exact online
softmax, causal triangular load balancing cho `S=100000`, `Dh=8` specialization
và batch–sequence 2D streaming nếu memory lại trở thành blocker. Nguồn code để
mine scheduling: FlashInfer SM120, FA4 SM120, Triton, CUTLASS 79, TileLang và
ThunderKittens; external microbenchmark không thay whole-model evidence.

## 16. Ma trận quyết định theo mục tiêu

| Mục tiêu | Candidate tốt nhất | Candidate thứ hai | “Idea độc lạ” | Tránh nhầm |
|---|---|---|---|---|
| Tốc độ attention exact | `NEXT`: FlashInfer SM120 | `SHIPPED` control: PyTorch Flash | causal persistent scheduler | cuDNN/FA4 đã thua exact measured path |
| Low memory exact | `SHIPPED`: Flash/online softmax + batch microchunk | `NEXT`: no-concat QKV, scratch reuse | 2D batch-query streaming | PagedAttention chủ yếu decode |
| Low precision error | `DEFERRED`: protected QK INT8 + PV FP16 | FP32 softmax/two-level accum | outlier correction path | Current Sage fail; kernel TOPS bỏ quant overhead |
| QKV | `NEXT`: TE LayerNormLinear + native layout | CUTLASS direct-layout epilogue | separate activation Q/K/V dù weight packed | Packed activation không luôn tốt #14 |
| FFN | `SHIPPED` exact pre-GELU control | `DEFERRED`: cuBLASLt/TE/CUTLASS exact fusion | fully resident D32 MLP | Tanh GELU và isolated win không đủ |
| GEMM | Workload router + Inductor autotune | CUTLASS SM120 | persistent grouped schedule | DeepGEMM SM100 không đồng nghĩa SM120 |
| Kernel optimize | Exact library backend first | Triton/CUTLASS sau library ceiling | ThunderKittens/TileLang | Microbench không thay end-to-end |
| CPU | oneDNN/IPEX | LIBXSMM/FBGEMM | SME2/KleidiAI | CPU compute không giúp timed CUDA |
| S=100000 innovation | FlashInfer exact + QKV/layout fusion | PyTorch Flash control | causal dynamic tile scheduler | Current Sage fail; sparse/linear đổi semantics |

## 17. Devil's advocate: những lý do report này có thể dẫn sai nếu dùng thiếu kỷ luật

1. **External benchmark không comparable**: khác GPU, dtype, shape, causal flag, head dim, warmup và metric.
2. **Kernel TOPS không phải layer speedup**: quantize, smooth, transpose, concat, projection và residual có thể lớn hơn phần kernel tiết kiệm.
3. **SM120 support đang chuyển động**: FA4 có file SM120 nhưng code hiện dùng SM80 MMA path; GitHub vẫn có issue package/varlen/GQA. Cần pin commit đã test.
4. **Vendor selector thay đổi theo version**: gọi cùng `scaled_dot_product_attention` có thể chạy backend khác sau upgrade.
5. **Approximate end-task quality không tương đương elementwise equality**: “lossless video/perplexity” vẫn có thể fail Track 3.
6. **Compiler có thể đổi peak memory**: CUDA Graph và max-autotune cache workspace; một mode nhanh ở #1 có thể OOM #14.
7. **Static library fusion có thể đổi math order**: exact về công thức nhưng accumulation/GELU/LayerNorm ordering đủ làm fail strict threshold.
8. **Baseline #14 là vấn đề benchmark protocol**: không thể lấy OOM làm một speedup số học. Cần report survival/memory và quy tắc organizer nếu chưa có baseline runnable.
9. **Code cũ có giá trị ý tưởng hơn giá trị drop-in**: FasterTransformer, LightSeq, Longformer và một số CUTLASS examples target CUDA/GPU cũ.
10. **Một global winner khó tồn tại**: 14 shapes cố tình quét batch, D, H và S; dispatch per shape là thiết kế hợp lý hơn.

## 18. Quy trình tìm kiếm và tiêu chí nguồn

### 18.1 Phạm vi tìm kiếm

Các truy vấn chính được xoay quanh:

- `FlashAttention 4 SM120`, `cuDNN fused attention Blackwell`, `FlashInfer SM120 prefill`;
- `SageAttention RTX 5090`, `quantized attention QK INT8 PV FP16`;
- `LayerNormLinear LayerNormMLP`, `cuBLASLt GELU epilogue`, `CUTLASS SM120 GEMM`;
- `Triton fused attention persistent matmul`, `TileLang attention`, `ThunderKittens Blackwell`;
- `long context exact attention ring ulysses burst`, `dynamic sparse prefill attention`;
- `oneDNN SDPA`, `LIBXSMM BRGEMM`, `FBGEMM`, `KleidiAI`.

### 18.2 Ưu tiên nguồn

1. Official repository/source code.
2. Official vendor documentation.
3. Paper gốc/arXiv/OpenReview/proceedings.
4. Issue chính chủ chỉ để ghi caveat hoặc tình trạng integration, không dùng làm bằng chứng performance tổng quát.

Không dùng blog tổng hợp/Reddit làm cơ sở cho shortlist. Các repo/paper architecture-changing vẫn được giữ vì user yêu cầu breadth và “idea độc lạ”, nhưng được gắn nhãn rõ.

### 18.3 Follow-up đã đo trên RTX 5090

Môi trường Vast.ai: RTX 5090, PyTorch `2.11.0+cu128`, CUDA 12.8. Profiler exact
inner executor V16 `B1/S100000/D1024/H16/L2` cho attention `199.0931 ms`, bằng
`92.258%` raw device time; đây là bằng chứng bottleneck, không phải ước lượng
từ complexity.

- Built-in sandwich: Flash `217.3188/218.4642 ms`, cuDNN `223.6575 ms`;
  Efficient `417.5209 ms`. Giữ Flash.
- FlashAttention-4 `4.0.0b28`: strict isolated attention PASS
  `0/102,400,000`, nhưng `108.7358 ms` chậm hơn PyTorch Flash `100.9365 ms`
  (`7.72%`). Reject theo performance.
- SageAttention commit `d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5`: automatic
  SM120 INT8-QK/FP8-PV fail accuracy mạnh. Recipe per-thread INT8-QK +
  PV-FP16/FP32-accum đạt `72.1337 ms` so Flash `100.5045 ms`, nhưng isolated
  strict FAIL `94/102.4M`.
- Version V18 đưa Sage recipe qua full Transformer B=1: eager strict FAIL
  `1/102.4M`, max abs `0.0026415`; compiled wrapper cũng chưa equivalent eager.
  Accuracy gate dừng trước model timing và không promote.
- Failure-locality rerun seed 1234 ghi `109/102.4M` isolated violations nhưng
  tất cả nằm ở query `1..31`; `minimal_exact_prefix=32`. V17-Sage vì vậy thử
  exact Flash prefix 32 + Sage suffix, FP32 attention out-projection và
  CUDA-Graph-unsafe custom op trên source-clean V16.1. Full #1–#13 GPU matrix
  fail strict ở #6/#9 dù #13 PASS và đạt `17.067x`; candidate bị reject.
- V18-Sage direct probe bỏ correction và gọi automatic SM120 INT8-QK/FP8-PV
  để đo raw performance. Đây là invalid diagnostic nếu accuracy fail và không
  đổi main V16.1.

Các số trên là exact-shape inner/isolated diagnostics hoặc B=1 accuracy
canary. Original baseline full #14 vẫn không executable trên 32 GiB, nên không
có paired official speedup. Artifacts nằm trong `runs/profiles/` và quy trình
đầy đủ ở `EXPERIMENTS.md` §17–18.

### 18.4 Những gì chưa làm

- Chưa probe FlashInfer SM120 trên exact workload.
- Chưa chạy FA4/Sage trên hardware hoặc PyTorch version thứ hai.
- V17-Sage selective-prefix correction fail strict #6/#9 và đã bị reject.
- V18-Sage direct automatic còn chờ custom-op `opcheck`, full timing matrix và
  shape-#14 optimized-only diagnostic trên target.
- Chưa có paired original-baseline latency/speedup cho shape #14 vì baseline
  score tensor cần khoảng `18.6 TiB`.

## 19. Checklist khi biến một mục trong report thành code

- [ ] Ghi hypothesis và source link vào `EXPERIMENTS.md`.
- [ ] Tạo file version mới, không overwrite implementation có kết quả hữu ích.
- [ ] Xác nhận API `forward(x, valid_token_mask)` và output shape.
- [ ] Map/copy toàn bộ weight; test state_dict/cache invalidation.
- [ ] Test causal/non-causal, mask/no-mask, dtype và nhiều shape.
- [ ] Accuracy gate strict OR trước benchmark.
- [ ] Log backend/kernel thực tế, không chỉ tên API wrapper.
- [ ] Đo layout/quantization/copy trong timed region.
- [ ] Chạy exact official shape với đúng GPU index mapping.
- [ ] Ghi baseline, optimized, latency, speedup, environment và command.
- [ ] Cập nhật `SOLUTION.md`, `IMPLEMENTATION_PLAN.md`, và `DECISION.md` nếu đổi hướng kiến trúc.

## 20. AI disclosure

Report này được tổng hợp với hỗ trợ của AI từ tài liệu repository và các nguồn công khai được liên kết trực tiếp. AI đã phân loại mức phù hợp và đề xuất các tổ hợp mới; các nhãn trạng thái/ưu tiên, dự đoán bottleneck và synthesis ở mục 14 là nhận định kỹ thuật, không phải kết quả thực nghiệm. Mọi quyết định promote implementation phải dựa trên accuracy/benchmark tái lập được theo luật của repository.
