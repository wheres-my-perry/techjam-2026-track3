# Frozen submission release

This page is the shortest path from a clean clone to the exact Perry V16.1
submission and its measured evidence.

## Canonical release identity

- Git tag: `techjam-2026-final-v16.1`
- Active entrypoint: `main.py`
- Standalone implementation: `v16_1_clean.py`
- Measured source revision: `4f77a04fd51c3a2ad8b9d8986657915ae3ca94d6`
- Machine-readable lock: `submission-manifest.json`

Resolve and verify the immutable release commit with:

```bash
git fetch --tags origin
git rev-list -n 1 techjam-2026-final-v16.1
git checkout --detach techjam-2026-final-v16.1
python3 tools/submission_preflight.py --require-clean --require-tag
```

The release commit contains later documentation and verification tooling, but
the active implementation and reference are byte-identical to the files used
for the measured run. The preflight enforces their SHA-256 hashes instead of
assuming that a later documentation commit changed performance.

## Locked source identity

| File | SHA-256 |
|---|---|
| `main.py` | `4c0807e93109c76584937be1a5f50db4276f34da9575d4270ab0d6cb6f28d672` |
| `v16_1_clean.py` | `522feff97b482e920d3dde542a659473bdc66ae04757205ab4b9b7c2e209025c` |
| `torch_transformer_benchmark.py` | `c072c48f22cb1438fe903c269eac9039c2554e0c247dcdbe147b9fe950af9500` |

The manifest also locks the dependency file, environment description, and two
curated final-result artifacts.

## Validated result

- Official shapes #1-#13: strict PASS, zero failed elements, worst maximum
  absolute error `0.00179085`, and predeclared same-host geometric-mean speedup
  `11.803x`.
- Official shape #14: full `B=32` strict PASS with `0/3,276,800,000` failed
  elements; optimized-only median `6987.4644 ms`, throughput `457,962.98`
  tokens/s, and peak allocated memory `24.487 GiB`.
- Shape #14 baseline latency and speedup remain N/A because the supplied
  reference would materialize an approximately `18.6 TiB` attention-score
  tensor.

The full commands and raw evidence are in [`results/final/`](results/final/README.md).

## Clean-clone CPU verification

Python 3.12 is recommended. The release lock can be checked before installing
PyTorch:

```bash
python3 tools/submission_preflight.py
```

After installing a CPU-compatible PyTorch build, run the same checks used by
CI:

```bash
python3 -m py_compile main.py torch_transformer_benchmark.py v16_1_clean.py \
  tools/submission_preflight.py tools/gpu_preflight.py
python3 -m tools.matrix_runner --list-shapes
python3 -m tools.profile_models --list-shapes
python3 -m unittest discover -s tests -v
```

These checks verify the active import, strict state-dict compatibility, mask
paths, exact 14-shape enumeration, locked hashes, and curated evidence schema.
They are not a substitute for the target-GPU benchmark.

## Target-GPU preflight

Install the final CUDA stack from `requirements.txt`, expose the intended GPU
as `cuda:0`, and run:

```bash
CUDA_VISIBLE_DEVICES=0 python3 tools/gpu_preflight.py \
  --device cuda:0 --strict-final-environment --require-idle
```

The GPU preflight first verifies the release lock, then checks CUDA, PyTorch,
Triton, device capability, memory, SDPA/compile availability, other compute
processes, and a strict baseline-versus-optimized CUDA correctness smoke. It
does not produce an official performance claim.

Use the longer commands in [`README.md`](README.md#steps-to-reproduce-the-results)
only after this preflight passes.

## Scope and licensing

The active artifact is frozen at V16.1. Files under `candidates/` and historical
versions under `archive/` are not part of the final implementation. Project-
authored work is licensed under [`LICENSE`](LICENSE); organizer materials and
optional dependencies are described in [`NOTICE.md`](NOTICE.md).
