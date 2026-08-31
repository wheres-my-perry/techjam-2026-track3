# Kế hoạch triển khai

## 1. Trạng thái tổng quan

**Phase hiện tại:** Phase 6 submission assembly — V16.1 đã được flatten vào
`v16_1_clean.py`; đây vẫn là active/final implementation. Root hiện có thêm
candidate versioned `candidates/v19/cuda_fp16_checkpoint.py`, nhưng V19 chưa qua CUDA gate và
không thay `main.py`.
File chứa đầy đủ model/config, FP16 cache, Flash-first attention, Triton
FP32-pre-GELU và compiled executor #14, không import harness hay version cũ.
`main.py` chỉ nối file standalone này vào benchmark harness; các version
`v1`–`v18` lịch sử đã chuyển sang `archive/versions/`. Thuật toán và số liệu
promotion D-038 không đổi. Fresh final run ngày 2026-08-31 đã validate đúng
standalone artifact trên RTX 5090 driver `595.71.05`: #1–#13 strict PASS với
predeclared start-control geomean `11.803x`, full #14 strict PASS
`0/3,276,800,000`, native B32 PASS và optimized-only median `6987.4644 ms`;
evidence được track trong `results/final/`. Full historical timeline cho 11
checkpoint và reverse-order repeat đã hoàn tất trong
`results/timeline/`. Evidence driver `580.159.03` cũ được
giữ trong `results/archive/cross-host-driver580/`; chênh `7.904x → 11.803x` không phải
code gain vì baseline host mới chậm hơn `72.48%`.

Trước cleanup, Phase 5 — V16.1 đã được promote qua `main.py` theo D-038:
#1–#13 dùng source-clean V14.1/V11 packed-QKV path, còn FP32 eval `S >= 8192`
chạy batch chunk với standalone compiled B=1 executor tái sử dụng. Promotion
ưu tiên bỏ exact official-#13 tuple/V15 dependency và chủ đích trả lại win
`0.98–2.20%` của V16 ở #13; không phải performance promotion. Predecessor V16
#14 PASS strict `0/3,276,800,000` và giảm optimized-only median `3.11–3.61%`
so với V14.1 trên RTX 5090/PyTorch 2.11. V16.1 khi đó mới có B=1 canary; full
#14 rerun trên artifact clean đã PASS ngày 2026-08-31, còn robustness nhiều
seed/hardware vẫn pending. V17 batch-chunk B=2 cũng PASS full strict nhưng chỉ giảm
`0.30–0.59%` so V16 trong alternating run, nên giữ experimental và không đổi
main V16 ở thời điểm D-033. Shape-#14 profiler sau đó xác nhận attention chiếm `92.258%`;
built-in cuDNN/Efficient và FA4 không thắng PyTorch Flash. V18 SageAttention có
isolated upside nhưng full-model B=1 vẫn fail strict `1/102.4M`, nên bị reject
theo D-034 và không benchmark model. V15.1 cross-shape sweep sau đó xác nhận
direct-layout QKV chỉ thắng thêm official #6 (`-3.43%` E2E, `-2.99%` raw GPU
theo geometric two-order ratio); D-035 giữ đây là candidate, chưa đổi main.
V16.1 source-clean artifact giữ compiled #14 nhưng bỏ hoàn toàn
V15/direct-QKV dependency; #13 và #14 B=1 canary PASS, và D-038 đã promote nó
làm main. D-039 sau đó mở candidate V17-Sage riêng: measured prefix locality
`P=32`, FP32 out-projection và no-CUDA-Graph integration đã được code. Full
#1–#13 GPU matrix fail strict ở #6/#9 nên candidate bị reject, main không đổi.
V18-Sage direct automatic hiện là performance-only probe trên V16.1; nó bỏ mọi
correction để đo raw Sage SM120 và không đủ điều kiện promotion nếu accuracy fail.

Repository đã có đề bài, benchmark Torch chính thức, chuỗi ablation đầy đủ và
active standalone artifact. Final GPU evidence đã bao phủ cả 14 official
shapes; #14 giữ baseline/speedup N/A đúng protocol vì original score tensor
không executable trên 32 GiB.

## 2. Quy ước trạng thái

- `[x]` Đã hoàn thành và có artifact/code.
- `[~]` Đang làm hoặc mới hoàn thành một phần.
- `[ ]` Chưa bắt đầu.
- `[!]` Bị chặn bởi thông tin/môi trường bên ngoài.

## 3. Phase 0 — Nắm đề và dựng khung tài liệu

- [x] Lưu Track 3 vào `STATEMENT.md`.
- [x] Lưu tên hai benchmark resource từ Lark.
- [x] Thêm `AGENTS.md`, `ARCHITECTURE.md`, `SOLUTION.md`, `EXPERIMENTS.md`, `IMPLEMENTATION_PLAN.md`, `DECISION.md`.
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
- [x] Thêm xuất kết quả máy đọc được (JSON/CSV) qua official matrix runner.
- [x] Thêm `tools/matrix_runner.py` chạy đúng 14-shape benchmark matrix, tiếp tục qua OOM/error/timeout.
- [x] Hiển thị `max_abs` trong bảng tổng kết terminal của matrix runner.
- [x] Thêm metadata môi trường: GPU capability, driver, CUDA, cuDNN, OS, CPU,
  RAM, disk và git revision trong `results/final/environment.json`.

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
- [x] Chốt policy: performance benchmark chỉ dùng đúng 14 test shapes chính thức; shape khác chỉ là diagnostic.
- [ ] Chạy accuracy matrix cho causal/non-causal và padding/no-padding.
- [ ] Chạy accuracy trên nhiều seed, input scale và tất cả shape công bố.
- [ ] Chạy benchmark GPU với warmup và nhiều round.
- [ ] Ghi kết quả, command và môi trường vào `EXPERIMENTS.md`.
- [ ] Xác nhận packed cache không stale sau mọi đường load/move/compile được dùng.

**Exit criteria:** mọi case mục tiêu pass accuracy và có kết quả speedup tái lập trên GPU mục tiêu.

## 6. Phase 3 — Profile và chọn bottleneck

- [x] Thêm `tools/profile_models.py` với accuracy gate, CUDA Event timing, PyTorch Profiler/Kineto, subprocess isolation và JSON/Chrome trace output.
- [~] Profile optimized path của v1/v2/v3 trên official shape #1 bằng PyTorch Profiler; baseline operator breakdown và Nsight còn pending.
- [x] Đã validate raw GPU kernel/device-event count, Triton/compiled-region/CUDA-Graph launch evidence cho compiled V3.1/V4/V4.1 trên official shape #1.
- [x] Tách category và non-overlapping model stages: QKV/output projection, attention core, từng LayerNorm, FFN in/GELU/out, residual, masking, copy và final norm.
- [~] Đã ghi peak extra CUDA allocation trong cửa sổ profile; chưa phân tích lifetime của từng intermediate.
- [ ] Xác định bottleneck theo từng nhóm shape.

**Exit criteria:** mỗi optimization tiếp theo có profiler evidence và target metric rõ ràng.

## 7. Phase 4 — Mở rộng optimization

- [x] Tách taxonomy: v2 = v1 + SDPA; chuyển packed-QKV no-copy + flattened whole-model loop sang `v3_SDPA_NoCopy.py`.
- [x] `v2_SPDA.py` PASS local causal/non-causal × padding/no-padding và GPU FP32 core shapes.
- [x] `v3_SDPA_NoCopy.py` GPU PASS và đạt 1.922x trên official shape #1; kết quả 1.781x non-causal chỉ là non-official diagnostic.
- [x] Dùng bản `v1_old` tạm thời để đối chiếu rồi xóa sau khi đã merge logic tốt hơn vào `v3_SDPA_NoCopy.py`.
- [x] V3.1 bỏ materialized causal mask và attention-level padding zero; GPU core branches PASS, official #1 đạt 2.152x và #13 đạt 7.697x.
- [~] `torch.compile(mode="reduce-overhead")` PASS official shape #1; V3.1 đạt `0.3168 ms`, còn V4 FP16/V4.1 đạt `0.1858 ms`; `default`/`max-autotune` và full matrix còn pending.
- [~] `v4_FP16.py` GPU official shape #1 PASS và eager đạt `0.5836 ms`; `v4_BF16.py` GPU accuracy FAIL nên không benchmark; full matrix FP16 pending.
- [~] `v4_1_FP16_GELU.py` bỏ tám FP16↔FP32 casts quanh GELU: official shape #1 eager `0.5492 ms`, compiled `0.1858 ms`; shapes #2/#7/#8/#12/#13 PASS, full matrix pending.
- [x] Tạo `v4_1_clean.py` standalone không phụ thuộc harness; strict state dict, eager output, training/BF16 fallback khớp exact V4.1; local và GPU compiled smoke PASS.
- [x] Sweep Flash/Efficient/cuDNN SDPA; tạo V4.2 static dispatcher, PASS #1–#13 và tăng geomean `7.09x → 7.58x` so với V4.1 matrix.
- [x] Bỏ key mask dư trên causal/right-padding shapes để mở PyTorch Flash; V4.3 PASS #1–#13 và tăng geomean `7.58x → 8.48x` so với V4.2.
- [x] Đơn giản hóa V4.3: bỏ static shape table, dùng Flash → cuDNN → Efficient → Math fallback; direct geomean đạt `8.52x`.
- [x] Tạo `v4_3_flash_clean.py` standalone; strict state dict, FP16 cache lifecycle, eager causal/non-causal và training/BF16 fallback khớp exact V4.3 benchmarkable trên local smoke.
- [x] Chạy V4.3 max-autotune #1–#13: strict PASS, geomean `9.5266x`, tăng `11.83%` so với reduce-overhead.
- [x] Probe FP8 trên RTX 5090 và triển khai per-tensor E4M3/MXFP8 block-scale;
  kernel sanity PASS nhưng full/single-scope model ablations đều FAIL accuracy,
  nên V5 bị loại trước performance benchmark.
- [x] Thử V5.1 full FP16 accumulation: max-autotune fail #10 và không nhanh hơn
  V4.3 ở paired #8; reduce-overhead fail #8, nên giữ làm negative ablation.
- [x] Thử V6 tanh-approximated GELU: strict accuracy PASS #1–#13, nhưng paired
  #2/#8 không có gain vượt noise và #2 tăng một compiled kernel; không promote.
- [x] Thử V7a residual + LayerNorm pipeline: correctness PASS, nhưng
  max-autotune sinh cùng kernel graph V4.3 trên #2/#8/#12; không promote và
  không viết V7b custom Triton vì compiler đã fuse mục tiêu.
- [x] Thử V8 FFN-in GEMM/bias/exact-GELU fusion: full #1–#13 accuracy PASS;
  official #6 giảm `26.6799 → 25.4092 ms` và `32 → 29` kernels. Dispatch chỉ
  bật cho `B*S >= 1M, D=FFN=128`; shape khác fallback V4.3.
- [x] Thử V8.1 force-all trên #1–#13: strict accuracy PASS, paired geomean
  latency giảm khoảng `0.97–1.36%`, nhưng #2/#12 order-sensitive và #11 raw
  GPU regression; giữ làm ablation, không thay dispatcher bằng unconditional.
- [x] Thử V9 fully fused persistent MLP: isolated kernel nhanh `1.18–1.59x` và
  #1–#13 accuracy PASS, nhưng whole-model D=128 regress, #7 đổi dấu, #6 chậm
  hơn V8; giữ làm ablation, không promote.
- [x] Thử V11 bỏ FP16 round trước exact GELU trong V8.1 custom epilogue: RTX
  5090 #1–#13 strict PASS. #6 giảm max abs `13.01%` với latency trung tính, #10
  giảm max abs `1.19%` và latency khoảng `1.50%`; #7 giảm error nhưng regress
  host `1.47–2.77%`. Theo D-027, chấp nhận trade-off và promote force-all V11
  làm main; shape #14 pending.
- [~] Thử V12 chỉ bỏ FP16 output round của FFN-out bằng CUDA FP32-output GEMM:
  local syntax/branch/state-dict/fallback/compile PASS và #7/#10 giảm mean error
  ở 10/10 seed. V12.1 attention-only và V12.2 cả hai boundary cũng PASS local;
  V12.2 giảm mean error nhiều nhất. GPU accuracy/paired latency pending; V11
  vẫn là arithmetic control dưới V15/V14.1 parent.
- [x] Thử V13 symmetric INT8 FFN-in accuracy probe: per-output-channel W8,
  per-token A8 và INT32 accumulation. Official #2 fail cả W8/A8/W8A8; W8-only
  layer cuối vẫn fail `69/81,920`, nên dừng trước GPU/kernel/performance.
- [x] Tạo V14 exact batch-chunk path cho #14: V11 full-batch OOM ở packed QKV
  `18.31 GiB`; V14 chunk `B=1` chạy full output với peak `28.526 GiB`. Strict
  memory-bounded reference PASS `0/3.2768B`, max abs `0.000831008`; optimized-only
  median `6683.9873 ms`. Baseline/speedup N/A vì explicit score ~`18.6 TiB`.
- [x] Hợp nhất thành V14.1: cutoff `S >= 8192`, FP32 eval và `B > 1` dùng
  batch-chunk; dưới cutoff/training/non-FP32/B=1 fallback V11. Large-sequence
  helper tắt Dynamo capture để tránh unroll loop #14; `main.py` đã promote.
- [x] V15 direct-layout QKV cho exact #13: Triton GEMM FP16/FP32-accumulator
  store thẳng `[3,B,H,S,Dh]` contiguous cho Flash. Official #13 strict PASS
  `0/41,943,040`; paired max-autotune hai orders giảm end-to-end `0.98–2.20%`
  và raw GPU `1.17–1.76%`. Padding/non-causal/fallback canaries PASS; D-031
  promote V15 qua `main.py`.
- [x] V16 compile/reuse riêng thân B=1 của batch-chunk #14, giữ loop ngoài
  eager. Full strict #14 PASS `0/3.2768B`, max abs `0.000944197`; sandwich
  V14.1→V16→V14.1 giảm median `3.11–3.61%` và peak `26.977 → 24.487 GiB`.
  D-032 từng promote V16; D-038 giữ nó làm direct-QKV rollback.
- [x] V17 thử compiled batch chunk B=2: full #14 strict PASS, timed peak giữ
  `24.487 GiB`; alternating V16/V17 giảm median trung bình `0.515%` nhưng chỉ
  `0.30–0.59%` theo adjacent controls. D-033 giữ V17 làm ablation và V16 làm
  main ở thời điểm đó.
- [x] Xác nhận SDPA là bottleneck #14 bằng inner profiler; shootout built-in,
  FA4 và Sage. V18 Sage strict FAIL nên giữ PyTorch Flash; D-038 chỉ đổi main
  wrapper, không đổi attention backend.
- [x] Force direct-layout QKV V15.1 trên official #1–#12: strict accuracy PASS
  toàn bộ; paired hai orders chỉ #6 thắng đồng thời host/raw device. Retest dài
  loại apparent wins #2/#4; D-035 không promote force-all và giữ main V16 ở
  thời điểm đó.
- [x] Làm sạch V16.1: kế thừa trực tiếp V14.1, chứa riêng compiled executor #14,
  không import/MRO qua V15 hay exact tuple. Official #13 và exact-config #14
  B=1 canary PASS; D-038 promote V16.1 làm main, V16 thành direct-QKV rollback.
- [x] Tạo và reject V17-Sage trên V16.1: Sage per-thread INT8-QK,
  PV-FP16/FP32-accum, exact causal prefix 32 theo measured failure locality,
  FP32 attention out-projection và CUDA-Graph-unsafe custom op. Aliases/harness
  đã được nối; full #1–#13 RTX 5090 matrix fail strict ở #6/#9 nên không đổi main.
- [~] V18-Sage direct automatic trên source-clean V16.1: SM120 dùng Sage
  INT8-QK/FP8-PV, bỏ prefix/FP32-out correction; #8 fallback do Dh=256. Đây là
  benchmark-on-failure diagnostic, không phải accuracy-valid candidate.
- [x] V19 thay riêng FFN-in/GELU của V16.1 bằng CUDA WMMA accumulator FP16 và
  checkpoint partial sum vào FP32 theo K. Default K=32; controls K=16/64/128 và
  WMMA-FP32 dùng cùng layout/epilogue. GPU K64 full #1–#14 PASS nhưng #1–#13
  geomean `10.3079x` so V16.1 `11.8030x`; reject performance, `main.py` vẫn
  V16.1.
- [x] V19.1.0/V19.1.1 chia large batch thành 1/2/4/8/16/32 partition cân bằng
  và enqueue trên nhiều CUDA stream; bản đầu kế thừa V16.1, bản sau kế thừa
  V19. GPU sweep chọn V19.1.0 P4: full #14 PASS `0/3.2768B`, median
  `6780.3867 ms`, peak `25.676 GiB`; P8 regress và P16 không chạy. V19.1.1
  K64/P2 PASS nhưng chậm hơn. `main.py` chưa đổi.
- [ ] So sánh code complexity, compile time, portability và speedup.

**Exit criteria:** có ít nhất một candidate tốt nhất cho mỗi nhóm shape quan trọng.

## 8. Phase 5 — Scheduler theo shape

- [x] Chốt initial dispatch key `(B,S,D,H,L,FFN,causal)` cho RTX 5090 FP32-public path.
- [x] Xây registry cuDNN cho #1/#2/#3/#4/#7/#9/#13; automatic SDPA là fallback.
- [x] Mở rộng registry V4.3: Flash cho #1/#4/#5/#7/#8/#10/#11/#13, cuDNN cho #2/#3/#9, automatic cho #6/#12/shape lạ.
- [x] Supersede registry trên bằng `v4_3_Flash.py` Flash-first; giữ static dispatcher như ablation lịch sử.
- [~] Chạy offline autotuning SDPA backend trên GPU mục tiêu; precision/kernel candidates khác còn pending.
- [x] Sinh bảng dispatch tĩnh và safe fallback cho shape chưa biết.
- [x] Thêm V8 FFN/GELU dispatch cho large-token D=FFN=128; giữ V4.3 cho các
  config mà paired whole-model benchmark không chứng minh gain.
- [x] Promote V16.1 làm main theo D-038; #1–#13 dùng packed-QKV V14.1/V11 và
  #14 dùng standalone compiled sample executor. V16 là rollback có direct-QKV
  #13; V15/V14.1/V11 là các rollback thấp hơn.
- [x] Kiểm tra scheduler không thêm tensor branch/host sync vào compiled hot path.
- [x] Có một source-clean main artifact thống nhất: V11 packed-QKV path cho
  #1–#13 và compiled chunked path #14, không hard-code official tuple.
- [~] Đã tìm được #6 là direct-QKV winner thứ hai bằng cross-shape sweep; chưa
  thêm dispatcher vì cần workload predicate/robustness gate thay cho exact
  test tuple.
- [x] Stable aliases `main`/`best` trỏ V16.1 source-clean; alias `v16` giữ
  direct-QKV #13 để rollback/ablate mà không mất executor #14.
- [x] Flatten V16.1 thành `v16_1_clean.py` standalone, chuyển `main`/shape-#14
  tools sang file này và archive toàn bộ version cũ khỏi root.

**Exit criteria:** scheduler không làm sai case lạ và cải thiện aggregate score so với một implementation duy nhất.

## 9. Phase 6 — Submission

- [x] Dọn repository: root chỉ giữ active entrypoints, candidate vào
  `candidates/`, runner vào `tools/`, supplemental docs vào `docs/`, curated
  evidence vào `results/`, generated artifacts vào gitignored `runs/`, và thêm
  import/state-dict/mask-path smoke tests. Dependency install manifest đã có
  trong `requirements.txt`; clone-clean GPU reproduction vẫn là close-out gate.
- [x] Viết README cài đặt, chạy và tái lập kết quả.
- [x] Chốt public repository.
- [x] Cập nhật `SOLUTION.md` bằng fresh final matrix, environment và raw evidence.
- [x] Chạy full 11-checkpoint timeline trên đủ #1–#13, V16.1 start/end drift
  control và reverse-order repeat cho các aggregate chênh dưới 3%.
- [x] Chạy shape #14 theo scope cuối Baseline/V16.1: static feasibility,
  streamed strict, native B32 và optimized-only timing.
- [~] Đã tạo `docs/DEVPOST.md` submission-ready; còn điền team contributions và
  public YouTube URL từ owner.
- [ ] Quay demo video public trên YouTube.
- [ ] Kiểm tra licensing/trademark/copyright.

**Exit criteria:** một reviewer mới có thể clone, chạy accuracy/benchmark và tái lập bảng kết quả.

## 10. Việc ưu tiên tiếp theo

### Submission close-out

1. Review Devpost description từ final driver-595 table trong `results/final/`,
   nêu rõ geomean #1–#13, cross-host caveat và tách #14 optimized-only khỏi
   paired speedup.
2. Chốt dependency installation manifest cho Python `3.12`, PyTorch
   `2.11.0+cu128` và Triton `3.6.0`; kiểm tra clone-clean reproduction.
3. Bổ sung team-member contributions, development tools/API disclosure và link
   public demo video — đây là dữ liệu owner-specific không thể suy từ benchmark.
4. Quay demo: clone/import standalone artifact, chạy một short-shape strict
   gate, trình bày final matrix và walkthrough shape-#14 memory-bounded path.
5. Kiểm tra license, trademark và copyrighted assets trước khi public video.

### Post-submission optimization roadmap

6. **Exact attention first:** viết adapter version hóa cho FlashInfer SM120,
   gate #13 rồi #14 B1, đưa layout/adapter copy vào timing và giữ PyTorch Flash
   làm control. Không rerun cuDNN/FA4/Sage hiện tại như candidate mới vì đã có
   performance/correctness evidence chặn promotion.
7. **QKV/layout/memory fusion:** thử TE `LayerNormLinear` hoặc CUTLASS/custom
   direct-layout epilogue; với #14 thử separate/no-concat QKV activation và
   scratch reuse. Gate từng thay đổi riêng trước khi compose.
8. **Accuracy-aware router:** bắt đầu từ measured direct-QKV winner cho large
   `B*S`, `D=FFN=128`; dùng workload/GPU predicate, full strict matrix,
   reverse-order/raw-device evidence và aggregate gate, không hard-code test ID.
9. **Robustness/portability:** mở rộng multi-seed, input scale, padding,
   causal/non-causal và mask coverage; log actual backend, peak
   allocated/reserved, compile cold-start và steady-state. Rerun trên
   GPU/software stack thứ hai; không thay start-control `11.803x` bằng một run
   đẹp hơn.
10. **Later exact kernels:** chỉ thử cuBLASLt/TE/CUTLASS FFN cho #6/#8 hoặc
    `D=32`/`Dh=8` persistent kernel cho #7/#11 sau target profiling.
11. **Deferred research:** protected low precision và custom SM120 attention chỉ
    mở khi exact library/fusion path đã được đo. Accuracy-only #1/#8/#13 đứng
    trước mọi timing low-precision.
12. Giữ V16.1 làm main cho tới quyết định promotion riêng. V19 checkpointed-FP16
    đã bị GPU performance gate reject. V19.1.0 P4 full strict PASS và thắng
    V16.1 P1 controls `1.51–1.66%` trên #14; giữ làm measured candidate nhưng
    không tự thay final evidence. FlashInfer/layout work vẫn có upside lớn hơn.
