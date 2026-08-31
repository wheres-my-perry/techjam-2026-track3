# Archived implementation versions

This directory preserves the historical `v1`–`v18` implementation and
opcheck files. They are retained as experiment evidence and rollback material,
but they are no longer active runner targets.

The only active implementation in the repository root is
`../../v16_1_clean.py`. It is standalone and does not import files from this
archive or the benchmark harness.

Historical files keep their original absolute sibling imports. To reproduce
one directly from the repository root, expose both the root and this directory
on `PYTHONPATH`, for example:

```bash
PYTHONPATH="$PWD:$PWD/archive/versions" \
  python archive/versions/v11_FP32PreGELU.py --help
```

Archived results and their original commands remain documented in
`../../EXPERIMENTS.md` and `../../SOLUTION.md`.
