# Third-party notices and scope of the project license

Except for the materials identified below, project-authored source code and
documentation in this repository are provided under the MIT License in
[`LICENSE`](LICENSE).

## Organizer-provided materials

- `torch_transformer_benchmark.py` originated from the TikTok TechJam 2026
  Track 3 organizer benchmark. This repository normalized its comparator to
  the published strict correctness rule and added local benchmarking support.
- `STATEMENT.md` is a local restatement of the TikTok TechJam 2026 Track 3
  problem statement and links to its source.

The repository's MIT License does not override any rights or terms that apply
to those organizer-provided materials. They are included for participation,
reproduction, and review of the competition submission.

## Runtime and research dependencies

The active implementation uses PyTorch and optionally Triton. Historical
research tools can optionally integrate SageAttention, FlashAttention-4, and
NVIDIA CUTLASS DSL. These packages are not vendored in this repository and
remain subject to their own licenses and notices.

PyTorch, Triton, CUDA, NVIDIA, TikTok, and other product names are the property
of their respective owners. Their names are used only to identify compatible
software, hardware, or the competition; no endorsement is implied.

## Data and media

The benchmark uses generated tensors and does not redistribute a dataset. No
demo-video music, stock media, logos, or other third-party media are included
in this repository. Any future demo must record its own asset permissions
separately.
