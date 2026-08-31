# Tools

Run repository tooling as Python modules from the repository root:

```bash
python3 -m tools.matrix_runner --list-shapes
python3 -m tools.profile_models --list-shapes
python3 -m tools.timeline_runner --list-checkpoints
python3 -m tools.shape14.timeline_runner --list-checkpoints
```

General matrix, profiling, and historical-timeline orchestration lives directly
under `tools/`. Memory-bounded official-shape-#14 utilities live under
`tools/shape14/`. Generated output defaults to the ignored `runs/` tree.
