#!/usr/bin/env python3
"""Run one implementation on the 14 official benchmark shapes."""

import argparse
import csv
import json
import os
import platform
import re
import shlex
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
IMPLEMENTATIONS = {
    "main": "main.py",
    "best": "main.py",
    "v16.1": "main.py",
    "v16_1": "main.py",
    "v16.1.clean": "main.py",
    "v16_1_clean": "main.py",
    "v19": "candidates/v19/cuda_fp16_checkpoint.py",
    "v19.cuda": "candidates/v19/cuda_fp16_checkpoint.py",
    "v19_cuda": "candidates/v19/cuda_fp16_checkpoint.py",
    "v19.1.0": "candidates/v19/parallel_batch_v161.py",
    "v19_1_0": "candidates/v19/parallel_batch_v161.py",
    "v19.1.1": "candidates/v19/parallel_batch_v19.py",
    "v19_1_1": "candidates/v19/parallel_batch_v19.py",
}


@dataclass(frozen=True)
class Shape:
    id: int
    batch_size: int
    d_model: int
    heads: int
    seq_len: int
    layers: int
    ffn_dim: int
    causal: bool = True


SHAPES = (
    Shape(1, 64, 128, 4, 128, 4, 128),
    Shape(2, 1, 128, 4, 128, 4, 128),
    Shape(3, 4, 128, 4, 128, 4, 128),
    Shape(4, 16, 128, 4, 128, 4, 128),
    Shape(5, 128, 128, 4, 128, 4, 128),
    Shape(6, 10000, 128, 4, 128, 4, 128),
    Shape(7, 64, 32, 4, 128, 4, 32),
    Shape(8, 64, 1024, 4, 128, 4, 1024),
    Shape(9, 64, 128, 1, 128, 4, 128),
    Shape(10, 64, 128, 2, 128, 4, 128),
    Shape(11, 64, 128, 16, 128, 4, 128),
    Shape(12, 64, 128, 4, 32, 4, 128),
    Shape(13, 64, 128, 4, 1024, 4, 128),
    Shape(14, 32, 1024, 16, 100000, 2, 1024),
)

NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
ACCURACY_RE = re.compile(
    rf"^summary: (PASS|FAIL) \| max_abs=({NUMBER}) \| max_rel=({NUMBER}) "
    r"\| failed=(\d+)/(\d+)$",
    re.MULTILINE,
)
SPEEDUP_RE = re.compile(rf"^speedup\s+: ({NUMBER})x", re.MULTILINE)
CSV_FIELDS = (
    "shape_id",
    "status",
    "batch_size",
    "seq_len",
    "d_model",
    "heads",
    "layers",
    "ffn_dim",
    "causal",
    "accuracy",
    "max_abs",
    "max_rel",
    "failed",
    "total",
    "baseline_median_ms",
    "baseline_p90_ms",
    "baseline_throughput",
    "optimized_median_ms",
    "optimized_p90_ms",
    "optimized_throughput",
    "speedup",
    "duration_s",
    "returncode",
    "error",
    "command",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run an implementation file on the 14 official Track 3 shapes"
    )
    parser.add_argument(
        "--impl",
        default="main",
        help=(
            "Python file path, relative to this repo or absolute; aliases: "
            "main, best, v16.1, v16_1, v16.1.clean, v16_1_clean, "
            "v19, v19.cuda, v19_cuda, v19.1.0, v19_1_0, "
            "v19.1.1, v19_1_1"
        ),
    )
    parser.add_argument(
        "--shape-ids",
        default="all",
        help="comma-separated official IDs, e.g. 1,2,7; default: all",
    )
    parser.add_argument("--list-shapes", action="store_true")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype", choices=("float32", "float16", "bfloat16"), default="float32"
    )
    parser.add_argument("--accuracy-trials", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--benchmark-rounds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--timeout", type=int, default=900, help="seconds per shape")
    parser.add_argument(
        "--matmul-precision", choices=("highest", "high", "medium"), default="high"
    )
    parser.add_argument("--no-allow-tf32", action="store_true")
    parser.add_argument("--compile-baseline", action="store_true")
    parser.add_argument("--compile-user", action="store_true")
    parser.add_argument(
        "--benchmark-on-failure",
        action="store_true",
        help=(
            "diagnostic only: time an implementation even when strict accuracy "
            "fails; status remains ACCURACY_FAIL"
        ),
    )
    parser.add_argument(
        "--compile-mode",
        choices=("default", "reduce-overhead", "max-autotune"),
        default="default",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "runs" / "benchmarks"
    )
    return parser.parse_args()


def selected_shapes(value: str) -> list[Shape]:
    if value == "all":
        return list(SHAPES)
    try:
        ids = {int(item) for item in value.split(",")}
    except ValueError as exc:
        raise ValueError("--shape-ids must be 'all' or comma-separated integers") from exc
    invalid = ids - {shape.id for shape in SHAPES}
    if not ids or invalid:
        raise ValueError(f"invalid official shape IDs: {sorted(invalid)}")
    return [shape for shape in SHAPES if shape.id in ids]


def print_shapes() -> None:
    print("ID      B       S     D   H   L   FFN  causal")
    for shape in SHAPES:
        print(
            f"{shape.id:>2} {shape.batch_size:>6} {shape.seq_len:>7} "
            f"{shape.d_model:>5} {shape.heads:>3} {shape.layers:>3} "
            f"{shape.ffn_dim:>5}  {str(shape.causal).lower()}"
        )


def resolve_implementation(value: str) -> Path:
    path = Path(IMPLEMENTATIONS.get(value, value)).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    path = path.resolve()
    if not path.is_file():
        raise ValueError(f"implementation not found: {path}")
    return path


def build_command(
    args: argparse.Namespace, shape: Shape, script: Path
) -> list[str]:
    command = [
        args.python,
        str(script),
        "--device",
        args.device,
        "--dtype",
        args.dtype,
        "--batch-size",
        str(shape.batch_size),
        "--seq-len",
        str(shape.seq_len),
        "--d-model",
        str(shape.d_model),
        "--heads",
        str(shape.heads),
        "--ffn-dim",
        str(shape.ffn_dim),
        "--layers",
        str(shape.layers),
        "--accuracy-trials",
        str(args.accuracy_trials),
        "--warmup",
        str(args.warmup),
        "--repeats",
        str(args.repeats),
        "--benchmark-rounds",
        str(args.benchmark_rounds),
        "--seed",
        str(args.seed),
        "--matmul-precision",
        args.matmul_precision,
    ]
    if shape.causal:
        command.append("--causal")
    if args.no_allow_tf32:
        command.append("--no-allow-tf32")
    if args.compile_baseline:
        command.append("--compile-baseline")
    if args.compile_user:
        command.append("--compile-user")
    if args.benchmark_on_failure:
        command.append("--benchmark-on-failure")
    if args.compile_baseline or args.compile_user:
        command.extend(("--compile-mode", args.compile_mode))
    return command


def timing(output: str, label: str) -> dict[str, float | None]:
    match = re.search(
        rf"^{label}\s*: median=({NUMBER}) ms \| mean={NUMBER} ms "
        rf"\| p90=({NUMBER}) ms \| min={NUMBER} ms "
        rf"\| throughput=({NUMBER}) token/s$",
        output,
        re.MULTILINE,
    )
    if not match:
        return {"median_ms": None, "p90_ms": None, "throughput": None}
    return {
        "median_ms": float(match.group(1)),
        "p90_ms": float(match.group(2)),
        "throughput": float(match.group(3)),
    }


def parse_result(shape: Shape, command: list[str], output: str, code, elapsed) -> dict:
    accuracy = ACCURACY_RE.search(output)
    baseline = timing(output, "baseline")
    optimized = timing(output, "optimized")
    speedup = SPEEDUP_RE.search(output)

    if code == "timeout":
        status = "TIMEOUT"
    elif accuracy and accuracy.group(1) == "FAIL":
        status = "ACCURACY_FAIL"
    elif code == 0 and accuracy and baseline["median_ms"] and optimized["median_ms"]:
        status = "PASS"
    else:
        status = "ERROR"

    last_line = next((line for line in reversed(output.splitlines()) if line.strip()), "")
    if status == "TIMEOUT":
        error = "shape timed out"
    elif status == "ACCURACY_FAIL":
        error = "accuracy check failed"
    elif status == "ERROR":
        error = f"returncode={code}: {last_line}".strip()
    else:
        error = None

    return {
        "shape_id": shape.id,
        **{key: value for key, value in asdict(shape).items() if key != "id"},
        "status": status,
        "accuracy": accuracy.group(1) if accuracy else None,
        "max_abs": float(accuracy.group(2)) if accuracy else None,
        "max_rel": float(accuracy.group(3)) if accuracy else None,
        "failed": int(accuracy.group(4)) if accuracy else None,
        "total": int(accuracy.group(5)) if accuracy else None,
        "baseline_median_ms": baseline["median_ms"],
        "baseline_p90_ms": baseline["p90_ms"],
        "baseline_throughput": baseline["throughput"],
        "optimized_median_ms": optimized["median_ms"],
        "optimized_p90_ms": optimized["p90_ms"],
        "optimized_throughput": optimized["throughput"],
        "speedup": float(speedup.group(1)) if speedup else None,
        "duration_s": round(elapsed, 3),
        "returncode": code if isinstance(code, int) else None,
        "error": error,
        "command": shlex.join(command),
        "output": output,
    }


def save_results(base: Path, metadata: dict, results: list[dict]) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    metadata["updated_at"] = datetime.now(timezone.utc).isoformat()
    base.with_suffix(".json").write_text(
        json.dumps({"metadata": metadata, "results": results}, indent=2),
        encoding="utf-8",
    )
    with base.with_suffix(".csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)


def git_revision() -> str | None:
    result = subprocess.run(
        ("git", "rev-parse", "HEAD"), cwd=ROOT, text=True, capture_output=True
    )
    return result.stdout.strip() if result.returncode == 0 else None


def main() -> int:
    args = parse_args()
    if args.list_shapes:
        print_shapes()
        return 0
    if min(args.accuracy_trials, args.repeats, args.benchmark_rounds) <= 0:
        raise SystemExit("accuracy-trials, repeats and benchmark-rounds must be positive")
    if args.warmup < 0 or args.timeout <= 0:
        raise SystemExit("warmup must be non-negative and timeout must be positive")
    try:
        shapes = selected_shapes(args.shape_ids)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    try:
        script = resolve_implementation(args.impl)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    base = args.output_dir / f"matrix_{script.stem}_{args.dtype}_{stamp}"
    metadata = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "implementation": script.name,
        "implementation_arg": args.impl,
        "script": str(script),
        "python": args.python,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "device": args.device,
        "dtype": args.dtype,
        "shape_ids": [shape.id for shape in shapes],
        "accuracy_trials": args.accuracy_trials,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "benchmark_rounds": args.benchmark_rounds,
        "seed": args.seed,
        "timeout": args.timeout,
        "matmul_precision": args.matmul_precision,
        "allow_tf32": not args.no_allow_tf32,
        "compile_baseline": args.compile_baseline,
        "compile_user": args.compile_user,
        "compile_mode": args.compile_mode,
        "benchmark_on_failure": args.benchmark_on_failure,
        "git_revision": git_revision(),
    }
    results = []

    print(f"implementation={script}, shapes={len(shapes)}, output={base}")
    if args.benchmark_on_failure:
        print(
            "[warning] --benchmark-on-failure enabled: timings for "
            "ACCURACY_FAIL rows are invalid diagnostics",
            flush=True,
        )
    for index, shape in enumerate(shapes, 1):
        command = build_command(args, shape, script)
        print(
            f"[{index:>2}/{len(shapes)}] shape #{shape.id}: "
            f"B={shape.batch_size} S={shape.seq_len} D={shape.d_model} "
            f"H={shape.heads} L={shape.layers} FFN={shape.ffn_dim}",
            flush=True,
        )
        started = time.monotonic()
        try:
            process = subprocess.run(
                command,
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=args.timeout,
            )
            output, code = process.stdout, process.returncode
        except subprocess.TimeoutExpired as exc:
            raw = exc.stdout or ""
            output = raw.decode() if isinstance(raw, bytes) else raw
            code = "timeout"
        except OSError as exc:
            output, code = str(exc), "spawn_error"

        result = parse_result(
            shape, command, output, code, time.monotonic() - started
        )
        results.append(result)
        save_results(base, metadata, results)

        speedup = f"{result['speedup']:.3f}x" if result["speedup"] else "-"
        print(f"      {result['status']}: speedup={speedup}")
        if result["status"] != "PASS":
            tail = "\n".join(output.splitlines()[-12:])
            if tail:
                print(tail)

    print("\nID  status           max_abs  baseline   optimized  speedup")
    for result in results:
        max_abs = result["max_abs"]
        baseline = result["baseline_median_ms"]
        optimized = result["optimized_median_ms"]
        speedup = result["speedup"]
        print(
            f"{result['shape_id']:>2}  {result['status']:<14} "
            f"{f'{max_abs:.6g}' if max_abs is not None else '-':>9}  "
            f"{baseline if baseline is not None else '-':>8}   "
            f"{optimized if optimized is not None else '-':>9}  "
            f"{f'{speedup:.3f}x' if speedup is not None else '-':>7}"
        )
    print(f"\nJSON: {base.with_suffix('.json')}")
    print(f"CSV : {base.with_suffix('.csv')}")
    return 0 if all(result["status"] == "PASS" for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
