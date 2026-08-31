#!/usr/bin/env python3
"""Run historical Transformer checkpoints under one benchmark protocol."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import matrix_runner
from timeline_adapter import (
    CHECKPOINTS,
    PRIMARY_CHECKPOINTS,
    checkpoint_manifest,
    print_checkpoints,
    resolve_checkpoint,
    sha256_file,
)


ROOT = Path(__file__).resolve().parent
ADAPTER = ROOT / "timeline_adapter.py"
CSV_FIELDS = ("checkpoint_id", "checkpoint_label", *matrix_runner.CSV_FIELDS)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark historical checkpoints on official Track 3 shapes"
    )
    parser.add_argument(
        "--checkpoints",
        default="primary",
        help="comma-separated checkpoint IDs, or 'primary'",
    )
    parser.add_argument(
        "--shape-ids",
        default="1-13",
        help="official shape IDs/ranges (default: 1-13); shape #14 uses its own runner",
    )
    parser.add_argument("--list-checkpoints", action="store_true")
    parser.add_argument("--list-shapes", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
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
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument(
        "--matmul-precision", choices=("highest", "high", "medium"), default="high"
    )
    parser.add_argument("--no-allow-tf32", action="store_true")
    parser.add_argument(
        "--compile-mode",
        choices=("default", "reduce-overhead", "max-autotune"),
        default="max-autotune",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "benchmark-results" / "timeline-rtx5090-driver595",
    )
    parser.add_argument("--run-id")
    parser.add_argument("--environment-id", default="unlocked")
    parser.add_argument("--source-revision")
    parser.add_argument(
        "--control-drift-threshold",
        type=float,
        default=0.03,
        help="maximum relative start/end drift for repeated checkpoint controls",
    )
    parser.add_argument(
        "--allow-busy-gpu",
        action="store_true",
        help="diagnostic override; official sweeps require an idle GPU",
    )
    return parser.parse_args(argv)


def selected_checkpoints(value: str) -> list[str]:
    if value == "primary":
        return list(PRIMARY_CHECKPOINTS)
    checkpoint_ids = [item.strip() for item in value.split(",") if item.strip()]
    if not checkpoint_ids:
        raise ValueError("--checkpoints cannot be empty")
    for checkpoint_id in checkpoint_ids:
        resolve_checkpoint(checkpoint_id)
    return checkpoint_ids


def selected_timeline_shapes(value: str) -> list[matrix_runner.Shape]:
    if value == "all":
        value = "1-13"
    ids: set[int] = set()
    try:
        for item in value.split(","):
            token = item.strip()
            if not token:
                continue
            if "-" in token:
                start_text, end_text = token.split("-", 1)
                start, end = int(start_text), int(end_text)
                if start > end:
                    raise ValueError
                ids.update(range(start, end + 1))
            else:
                ids.add(int(token))
    except ValueError as exc:
        raise ValueError(
            "--shape-ids must contain official IDs/ranges such as '1-13'"
        ) from exc
    invalid = ids - set(range(1, 14))
    if not ids or invalid:
        raise ValueError(
            "timeline_runner supports only shapes #1-#13; use "
            f"shape14_timeline_runner.py for #14 (invalid: {sorted(invalid)})"
        )
    return [shape for shape in matrix_runner.SHAPES if shape.id in ids]


def command_output(command: Sequence[str]) -> dict:
    try:
        process = subprocess.run(
            command,
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"command": shlex.join(command), "returncode": None, "output": str(exc)}
    return {
        "command": shlex.join(command),
        "returncode": process.returncode,
        "output": process.stdout.strip(),
    }


def git_revision() -> str | None:
    result = command_output(("git", "rev-parse", "HEAD"))
    return result["output"] if result["returncode"] == 0 else None


def runtime_inventory(python: str) -> dict:
    code = (
        "import json, platform, torch; "
        "d={'python':platform.python_version(),'pytorch':torch.__version__,"
        "'cuda_wheel':torch.version.cuda,'cudnn':str(torch.backends.cudnn.version())}; "
        "\ntry:\n import triton; d['triton']=triton.__version__\n"
        "except Exception as e:\n d['triton']=None; d['triton_error']=str(e)\n"
        "print(json.dumps(d, sort_keys=True))"
    )
    result = command_output((python, "-c", code))
    if result["returncode"] == 0:
        try:
            return json.loads(result["output"].splitlines()[-1])
        except (IndexError, json.JSONDecodeError):
            pass
    return {"error": result}


def gpu_inventory() -> dict:
    inventory = command_output(
        (
            "nvidia-smi",
            "--query-gpu=name,uuid,driver_version,memory.total,pstate,temperature.gpu,clocks.sm,power.draw",
            "--format=csv,noheader,nounits",
        )
    )
    processes = command_output(
        (
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        )
    )
    return {"inventory": inventory, "compute_processes": processes}


def gpu_is_idle(gpu: dict) -> bool:
    processes = gpu["compute_processes"]
    return processes["returncode"] == 0 and not processes["output"].strip()


def build_benchmark_command(args, checkpoint_id: str, shape) -> list[str]:
    spec = resolve_checkpoint(checkpoint_id)
    command = [
        args.python,
        str(ADAPTER),
        "--checkpoint",
        checkpoint_id,
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
    if spec.compile_user:
        command.extend(("--compile-user", "--compile-mode", args.compile_mode))
    return command


def run_preflight(args, checkpoint_id: str) -> dict:
    command = (
        args.python,
        str(ADAPTER),
        "--checkpoint",
        checkpoint_id,
        "--preflight-only",
    )
    started = time.monotonic()
    result = command_output(command)
    result["duration_s"] = round(time.monotonic() - started, 3)
    result["status"] = "PASS" if result["returncode"] == 0 else "ERROR"
    prefix = "TIMELINE_PREFLIGHT_JSON="
    for line in result["output"].splitlines():
        if line.startswith(prefix):
            try:
                result["details"] = json.loads(line[len(prefix) :])
            except json.JSONDecodeError:
                pass
    if result["status"] == "PASS" and "details" not in result:
        result["status"] = "ERROR"
        result["error"] = "preflight JSON marker missing"
    return result


def checkpoint_summary(
    checkpoint_id: str, results: list[dict], run_label: str | None = None
) -> dict:
    passed = [result for result in results if result["status"] == "PASS"]
    speedups = [result["speedup"] for result in passed if result["speedup"]]
    optimized = [
        result["optimized_median_ms"]
        for result in passed
        if result["optimized_median_ms"]
    ]
    complete = len(results) == 13 and len(passed) == 13
    measured_geomean = (
        math.exp(sum(math.log(value) for value in speedups) / len(speedups))
        if speedups
        else None
    )
    return {
        "checkpoint_id": checkpoint_id,
        "run_label": run_label or checkpoint_id,
        "status": (
            "NO_RESULTS"
            if not results
            else "PASS"
            if len(passed) == len(results)
            else "INCOMPLETE"
        ),
        "passed_shapes": len(passed),
        "total_shapes": len(results),
        "failed_elements": sum(result["failed"] or 0 for result in results),
        "worst_max_abs": max(
            (result["max_abs"] for result in results if result["max_abs"] is not None),
            default=None,
        ),
        "measured_speedup_geomean": measured_geomean,
        "official_full_speedup_geomean": measured_geomean if complete else None,
        "optimized_geomean_ms": (
            math.exp(sum(math.log(value) for value in optimized) / len(optimized))
            if optimized
            else None
        ),
    }


def save_checkpoint(base: Path, metadata: dict, results: list[dict]) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": {**metadata, "updated_at": datetime.now(timezone.utc).isoformat()},
        "summary": checkpoint_summary(
            metadata["checkpoint_id"], results, metadata.get("run_label")
        ),
        "results": results,
    }
    base.with_suffix(".json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    with base.with_suffix(".csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)


def validate_args(args) -> tuple[list[str], list]:
    checkpoint_ids = selected_checkpoints(args.checkpoints)
    shapes = selected_timeline_shapes(args.shape_ids)
    if min(args.accuracy_trials, args.repeats, args.benchmark_rounds) <= 0:
        raise ValueError("accuracy trials, repeats, and rounds must be positive")
    if args.warmup < 0 or args.timeout <= 0:
        raise ValueError("warmup must be non-negative and timeout must be positive")
    if args.control_drift_threshold < 0:
        raise ValueError("control drift threshold must be non-negative")
    return checkpoint_ids, shapes


def relative_drift(start: float | None, end: float | None) -> float | None:
    if start is None or end is None or start == 0:
        return None
    return abs(end / start - 1.0)


def control_drift_report(
    checkpoint_runs: list[dict], threshold: float
) -> dict | None:
    by_checkpoint: dict[str, list[dict]] = {}
    for run in checkpoint_runs:
        by_checkpoint.setdefault(run["checkpoint_id"], []).append(run)
    repeated = {key: value for key, value in by_checkpoint.items() if len(value) >= 2}
    if not repeated:
        return None

    controls = []
    overall_pass = True
    for checkpoint_id, runs in repeated.items():
        start, end = runs[0], runs[-1]
        start_rows = {row["shape_id"]: row for row in start["results"]}
        end_rows = {row["shape_id"]: row for row in end["results"]}
        metrics = {}
        for metric in ("baseline_median_ms", "optimized_median_ms"):
            start_values = [
                row[metric]
                for row in start["results"]
                if row["status"] == "PASS" and row[metric] is not None
            ]
            end_values = [
                row[metric]
                for row in end["results"]
                if row["status"] == "PASS" and row[metric] is not None
            ]
            start_geomean = (
                math.exp(sum(math.log(value) for value in start_values) / len(start_values))
                if len(start_values) == len(start["results"]) and start_values
                else None
            )
            end_geomean = (
                math.exp(sum(math.log(value) for value in end_values) / len(end_values))
                if len(end_values) == len(end["results"]) and end_values
                else None
            )
            drift = relative_drift(start_geomean, end_geomean)
            metrics[f"{metric}_geomean"] = {
                "start": start_geomean,
                "end": end_geomean,
                "relative_drift": drift,
                "pass": drift is not None and drift <= threshold,
            }
        heavy = {}
        for shape_id in (6, 8, 13):
            shape_metrics = {}
            for metric in ("baseline_median_ms", "optimized_median_ms"):
                start_value = start_rows.get(shape_id, {}).get(metric)
                end_value = end_rows.get(shape_id, {}).get(metric)
                drift = relative_drift(start_value, end_value)
                shape_metrics[metric] = {
                    "start": start_value,
                    "end": end_value,
                    "relative_drift": drift,
                    "pass": drift is not None and drift <= threshold,
                }
            heavy[str(shape_id)] = shape_metrics
        control_pass = all(
            item["pass"] for item in metrics.values()
        ) and all(
            item["pass"]
            for shape_metrics in heavy.values()
            for item in shape_metrics.values()
        )
        overall_pass &= control_pass
        controls.append(
            {
                "checkpoint_id": checkpoint_id,
                "start_run_label": start["run_label"],
                "end_run_label": end["run_label"],
                "threshold": threshold,
                "metrics": metrics,
                "heavy_shapes": heavy,
                "status": "PASS" if control_pass else "FAIL",
            }
        )
    return {"status": "PASS" if overall_pass else "FAIL", "controls": controls}


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.list_checkpoints:
        print_checkpoints()
        return 0
    if args.list_shapes:
        print("ID      B       S     D   H   L   FFN  causal")
        for shape in matrix_runner.SHAPES:
            if shape.id == 14:
                continue
            print(
                f"{shape.id:>2} {shape.batch_size:>6} {shape.seq_len:>7} "
                f"{shape.d_model:>5} {shape.heads:>3} {shape.layers:>3} "
                f"{shape.ffn_dim:>5}  {str(shape.causal).lower()}"
            )
        return 0
    try:
        checkpoint_ids, shapes = validate_args(args)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = args.output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    actual_revision = git_revision()
    if (
        args.source_revision
        and actual_revision is not None
        and actual_revision != args.source_revision
    ):
        raise SystemExit(
            f"source revision mismatch: expected {args.source_revision}, got {actual_revision}"
        )
    revision = args.source_revision or actual_revision
    revision_verification = (
        "git-head"
        if actual_revision is not None
        else "declared-snapshot-with-file-sha256"
    )
    environment = {
        "environment_id": args.environment_id,
        "platform": platform.platform(),
        "python_executable": args.python,
        "runtime": runtime_inventory(args.python),
        "gpu_at_start": gpu_inventory(),
    }
    run_metadata = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "source_revision": revision,
        "source_revision_verification": revision_verification,
        "runner_sha256": sha256_file(Path(__file__)),
        "adapter_sha256": sha256_file(ADAPTER),
        "environment": environment,
        "checkpoint_ids": checkpoint_ids,
        "shape_ids": [shape.id for shape in shapes],
        "protocol": {
            "device": args.device,
            "dtype": args.dtype,
            "accuracy_trials": args.accuracy_trials,
            "warmup": args.warmup,
            "repeats": args.repeats,
            "benchmark_rounds": args.benchmark_rounds,
            "seed": args.seed,
            "timeout": args.timeout,
            "matmul_precision": args.matmul_precision,
            "allow_tf32": not args.no_allow_tf32,
            "compile_mode": args.compile_mode,
        },
        "invocation": shlex.join([sys.executable, *sys.argv]),
    }
    (run_dir / "run_metadata.json").write_text(
        json.dumps(run_metadata, indent=2), encoding="utf-8"
    )

    all_summaries = []
    checkpoint_runs: list[dict] = []
    occurrence_totals = {
        checkpoint_id: checkpoint_ids.count(checkpoint_id)
        for checkpoint_id in set(checkpoint_ids)
    }
    occurrence_seen: dict[str, int] = {}
    overall_ok = True
    for checkpoint_index, checkpoint_id in enumerate(checkpoint_ids, 1):
        spec = resolve_checkpoint(checkpoint_id)
        occurrence_seen[checkpoint_id] = occurrence_seen.get(checkpoint_id, 0) + 1
        occurrence = occurrence_seen[checkpoint_id]
        run_label = (
            f"{checkpoint_id}__{occurrence:02d}"
            if occurrence_totals[checkpoint_id] > 1
            else checkpoint_id
        )
        print(
            f"[{checkpoint_index}/{len(checkpoint_ids)}] {checkpoint_id}: {spec.label}",
            flush=True,
        )
        preflight = run_preflight(args, checkpoint_id)
        checkpoint_metadata = {
            **run_metadata,
            "checkpoint_id": checkpoint_id,
            "run_label": run_label,
            "occurrence": occurrence,
            "checkpoint": checkpoint_manifest(spec),
            "preflight": preflight,
        }
        base = run_dir / f"matrix_{run_label}_{args.dtype}"
        if preflight["status"] != "PASS":
            overall_ok = False
            save_checkpoint(base, checkpoint_metadata, [])
            all_summaries.append(
                checkpoint_summary(checkpoint_id, [], run_label)
            )
            print("      ERROR: checkpoint preflight failed", flush=True)
            continue
        if args.preflight_only:
            save_checkpoint(base, checkpoint_metadata, [])
            preflight_summary = checkpoint_summary(checkpoint_id, [], run_label)
            preflight_summary["status"] = "PREFLIGHT_PASS"
            all_summaries.append(preflight_summary)
            print("      PASS: import/state-dict preflight", flush=True)
            continue

        gpu_before = gpu_inventory()
        checkpoint_metadata["gpu_before"] = gpu_before
        if not args.allow_busy_gpu and not gpu_is_idle(gpu_before):
            overall_ok = False
            checkpoint_metadata["gpu_idle_gate"] = "FAIL"
            save_checkpoint(base, checkpoint_metadata, [])
            all_summaries.append(
                checkpoint_summary(checkpoint_id, [], run_label)
            )
            print("      ERROR: GPU has another compute process", flush=True)
            continue
        checkpoint_metadata["gpu_idle_gate"] = "PASS"

        results: list[dict] = []
        for shape_index, shape in enumerate(shapes, 1):
            command = build_benchmark_command(args, checkpoint_id, shape)
            print(
                f"      [{shape_index}/{len(shapes)}] shape #{shape.id}", flush=True
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

            result = matrix_runner.parse_result(
                shape, command, output, code, time.monotonic() - started
            )
            result["checkpoint_id"] = checkpoint_id
            result["checkpoint_label"] = spec.label
            results.append(result)
            save_checkpoint(base, checkpoint_metadata, results)
            speedup = f"{result['speedup']:.3f}x" if result["speedup"] else "-"
            print(f"          {result['status']}: {speedup}", flush=True)
            if result["status"] != "PASS":
                overall_ok = False
                tail = "\n".join(output.splitlines()[-10:])
                if tail:
                    print(tail, flush=True)

        summary = checkpoint_summary(checkpoint_id, results, run_label)
        all_summaries.append(summary)
        checkpoint_runs.append(
            {
                "checkpoint_id": checkpoint_id,
                "run_label": run_label,
                "results": results,
            }
        )
        print(
            f"      {summary['status']}: {summary['passed_shapes']}/"
            f"{summary['total_shapes']} shapes",
            flush=True,
        )

    drift_report = control_drift_report(
        checkpoint_runs, args.control_drift_threshold
    )
    if drift_report is not None:
        (run_dir / "control_drift.json").write_text(
            json.dumps(drift_report, indent=2), encoding="utf-8"
        )
        if drift_report["status"] != "PASS":
            overall_ok = False
    final_payload = {
        "metadata": {
            **run_metadata,
            "finished_at": datetime.now(timezone.utc).isoformat(),
        },
        "summaries": all_summaries,
        "control_drift": drift_report,
    }
    (run_dir / "timeline_summary.json").write_text(
        json.dumps(final_payload, indent=2), encoding="utf-8"
    )
    print(f"Artifacts: {run_dir}")
    return 0 if overall_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
