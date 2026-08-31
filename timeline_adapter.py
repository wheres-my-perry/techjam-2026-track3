#!/usr/bin/env python3
"""Load an archived checkpoint into the current official benchmark harness.

Every invocation is a fresh process.  This is important because archived modules
assign ``torch_transformer_benchmark.UserOptimizedTransformer`` at import time.
The adapter supplies only the implementation class; accuracy, weight copying,
input generation, and timing remain owned by ``torch_transformer_benchmark``.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parent
ARCHIVE = ROOT / "archive" / "versions"


@dataclass(frozen=True)
class CheckpointSpec:
    checkpoint_id: str
    label: str
    source: str
    class_name: str = "UserOptimizedTransformer"
    compile_user: bool = False
    dependencies: tuple[str, ...] = ()
    shape14_strategy: str = "eager"
    shape14_static_reason: str | None = None


_BENCH = "torch_transformer_benchmark.py"
_MIXED = "archive/versions/v4_mixed_precision_common.py"
_V43 = "archive/versions/v4_3_Flash.py"
_V8 = "archive/versions/v8_FusedFFNGELU.py"
_V11 = "archive/versions/v11_FP32PreGELU.py"
_V141 = "archive/versions/v14_1_BatchChunked.py"
_V15 = "archive/versions/v15_DirectQKVLayout.py"


CHECKPOINTS: dict[str, CheckpointSpec] = {
    "baseline": CheckpointSpec(
        "baseline",
        "PyTorch baseline",
        _BENCH,
        class_name="BaselineTransformer",
        shape14_strategy="static_infeasible",
        shape14_static_reason="explicit attention materializes an S-by-S score tensor",
    ),
    "v1": CheckpointSpec(
        "v1",
        "V1 fused QKV",
        "archive/versions/v1_fuseQKV.py",
        shape14_strategy="static_infeasible",
        shape14_static_reason="explicit attention materializes an S-by-S score tensor",
    ),
    "v2": CheckpointSpec("v2", "V2 SDPA", "archive/versions/v2_SPDA.py"),
    "v3_1_eager": CheckpointSpec(
        "v3_1_eager",
        "V3.1 eager",
        "archive/versions/v3_1_CausalMask.py",
    ),
    "v3_1_compiled": CheckpointSpec(
        "v3_1_compiled",
        "V3.1 compiled",
        "archive/versions/v3_1_CausalMask.py",
        compile_user=True,
        shape14_strategy="outer_compile",
    ),
    "v4_1": CheckpointSpec(
        "v4_1",
        "V4.1 mixed FP16",
        "archive/versions/v4_1_FP16_GELU.py",
        compile_user=True,
        dependencies=(_MIXED,),
        shape14_strategy="outer_compile",
    ),
    "v4_2": CheckpointSpec(
        "v4_2",
        "V4.2 SDPA dispatch",
        "archive/versions/v4_2_SDPA_Dispatch.py",
        compile_user=True,
        dependencies=(_MIXED,),
        shape14_strategy="outer_compile",
    ),
    "v4_3": CheckpointSpec(
        "v4_3",
        "V4.3 Flash-first",
        _V43,
        compile_user=True,
        dependencies=(_MIXED,),
        shape14_strategy="outer_compile",
    ),
    "v8": CheckpointSpec(
        "v8",
        "V8 fused FFN/GELU",
        _V8,
        compile_user=True,
        dependencies=(_V43, _MIXED),
        shape14_strategy="outer_compile",
    ),
    "v11": CheckpointSpec(
        "v11",
        "V11 FP32 pre-GELU",
        _V11,
        compile_user=True,
        dependencies=(_V8, _V43, _MIXED),
        shape14_strategy="outer_compile",
    ),
    "v15": CheckpointSpec(
        "v15",
        "V15 direct-layout QKV",
        _V15,
        compile_user=True,
        dependencies=(_V141, _V11, _V8, _V43, _MIXED),
    ),
    "v16": CheckpointSpec(
        "v16",
        "V16 compiled batch executor",
        "archive/versions/v16_CompiledBatchExecutor.py",
        compile_user=True,
        dependencies=(_V15, _V141, _V11, _V8, _V43, _MIXED),
        shape14_strategy="inner_executor",
    ),
    "v16_1": CheckpointSpec(
        "v16_1",
        "V16.1 standalone final",
        "v16_1_clean.py",
        compile_user=True,
        shape14_strategy="inner_executor",
    ),
}

PRIMARY_CHECKPOINTS = (
    "baseline",
    "v1",
    "v2",
    "v3_1_eager",
    "v3_1_compiled",
    "v4_1",
    "v4_2",
    "v4_3",
    "v8",
    "v11",
    "v16_1",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_files(spec: CheckpointSpec) -> tuple[Path, ...]:
    relative_paths = (spec.source, _BENCH, *spec.dependencies)
    paths: list[Path] = []
    seen: set[Path] = set()
    for relative in relative_paths:
        path = (ROOT / relative).resolve()
        if path not in seen:
            paths.append(path)
            seen.add(path)
    return tuple(paths)


def checkpoint_manifest(spec: CheckpointSpec) -> dict:
    files = []
    for path in checkpoint_files(spec):
        if not path.is_file():
            raise FileNotFoundError(f"checkpoint dependency not found: {path}")
        files.append(
            {
                "path": str(path.relative_to(ROOT)),
                "sha256": sha256_file(path),
            }
        )
    return {
        "checkpoint_id": spec.checkpoint_id,
        "label": spec.label,
        "source": spec.source,
        "class_name": spec.class_name,
        "compile_user": spec.compile_user,
        "shape14_strategy": spec.shape14_strategy,
        "shape14_static_reason": spec.shape14_static_reason,
        "files": files,
    }


def resolve_checkpoint(checkpoint_id: str) -> CheckpointSpec:
    try:
        return CHECKPOINTS[checkpoint_id]
    except KeyError as exc:
        choices = ", ".join(CHECKPOINTS)
        raise ValueError(
            f"unknown checkpoint {checkpoint_id!r}; choose one of: {choices}"
        ) from exc


def load_checkpoint(spec: CheckpointSpec):
    for path in (str(ROOT), str(ARCHIVE)):
        if path not in sys.path:
            sys.path.insert(0, path)

    import torch_transformer_benchmark as bench

    if spec.checkpoint_id == "baseline":
        implementation_class = bench.BaselineTransformer
        module_name = bench.__name__
    else:
        source = (ROOT / spec.source).resolve()
        module_name = f"_timeline_{spec.checkpoint_id}_{os.getpid()}"
        module_spec = importlib.util.spec_from_file_location(module_name, source)
        if module_spec is None or module_spec.loader is None:
            raise RuntimeError(f"cannot load checkpoint module: {source}")
        module = importlib.util.module_from_spec(module_spec)
        sys.modules[module_name] = module
        module_spec.loader.exec_module(module)
        implementation_class = getattr(module, spec.class_name)

    bench.UserOptimizedTransformer = implementation_class
    return bench, implementation_class, module_name


def run_preflight(spec: CheckpointSpec) -> dict:
    import torch

    bench, implementation_class, module_name = load_checkpoint(spec)
    config = bench.TransformerConfig(
        batch_size=1,
        seq_len=128,
        d_model=128,
        num_heads=4,
        ffn_dim=128,
        num_layers=4,
        causal=True,
    )
    baseline = bench.BaselineTransformer(config)
    optimized = implementation_class(config)
    bench.copy_model_weights(baseline, optimized, strict=True)

    baseline_state = baseline.state_dict()
    optimized_state = optimized.state_dict()
    if baseline_state.keys() != optimized_state.keys():
        raise RuntimeError("strict load passed but state_dict keys differ")
    unequal = [
        key
        for key, value in baseline_state.items()
        if not torch.equal(value, optimized_state[key])
    ]
    if unequal:
        raise RuntimeError(f"weight equivalence failed for: {unequal[:8]}")

    parameters = list(inspect.signature(implementation_class.forward).parameters)
    if parameters[:3] != ["self", "x", "valid_token_mask"]:
        raise RuntimeError(
            "forward signature must begin with (self, x, valid_token_mask); "
            f"got {parameters}"
        )

    return {
        "status": "PASS",
        "checkpoint": checkpoint_manifest(spec),
        "resolved_module": module_name,
        "resolved_class": (
            f"{implementation_class.__module__}.{implementation_class.__qualname__}"
        ),
        "forward_parameters": parameters,
        "state_dict_keys": len(baseline_state),
        "weight_equivalence": True,
    }


def print_checkpoints() -> None:
    print("checkpoint       compile  shape14             source")
    for checkpoint_id, spec in CHECKPOINTS.items():
        print(
            f"{checkpoint_id:<16} {str(spec.compile_user).lower():<7}  "
            f"{spec.shape14_strategy:<19} "
            f"{spec.source}"
        )


def parse_args(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser(
        description="Inject one timeline checkpoint into the official benchmark"
    )
    parser.add_argument("--checkpoint", choices=tuple(CHECKPOINTS))
    parser.add_argument("--list-checkpoints", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_known_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args, benchmark_args = parse_args(argv)
    if args.list_checkpoints:
        print_checkpoints()
        return 0
    if not args.checkpoint:
        raise SystemExit("--checkpoint is required")

    spec = resolve_checkpoint(args.checkpoint)
    if args.preflight_only:
        result = run_preflight(spec)
        print("TIMELINE_PREFLIGHT_JSON=" + json.dumps(result, sort_keys=True))
        return 0

    bench, _, _ = load_checkpoint(spec)
    sys.argv = [sys.argv[0], *benchmark_args]
    return bench.main()


if __name__ == "__main__":
    raise SystemExit(main())
