#!/usr/bin/env python3
"""Orchestrate isolated shape-#14 gates across historical checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence


_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.timeline_adapter import (
    CHECKPOINTS,
    PRIMARY_CHECKPOINTS,
    checkpoint_manifest,
    resolve_checkpoint,
    sha256_file,
)
from tools.timeline_runner import git_revision, gpu_inventory, gpu_is_idle, runtime_inventory


ROOT = Path(__file__).resolve().parents[2]
WORKER = Path(__file__).resolve().with_name("checkpoint_worker.py")
ADAPTER = ROOT / "tools" / "timeline_adapter.py"
RESULT_MARKER = "SHAPE14_STAGE_JSON="
STAGE_ORDER = (
    "b1-accuracy",
    "streamed-b32-accuracy",
    "native-b32-probe",
    "native-timing",
)
CSV_FIELDS = (
    "checkpoint_id",
    "checkpoint_label",
    "strategy",
    "b1_strict",
    "streamed_b32_strict",
    "native_b32",
    "timing",
    "failed",
    "total",
    "max_abs",
    "max_rel",
    "optimized_median_ms",
    "optimized_mean_ms",
    "optimized_p90_ms",
    "throughput_tokens_per_s",
    "peak_allocated_gib",
    "reason",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run shape #14 gates by checkpoint")
    parser.add_argument(
        "--checkpoints",
        default="baseline,v16_1",
        help="comma-separated IDs (default/final scope: baseline,v16_1)",
    )
    parser.add_argument("--list-checkpoints", action="store_true")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--batch-limit", type=int, default=32)
    parser.add_argument("--query-chunk", type=int, default=256)
    parser.add_argument("--compare-token-chunk", type=int, default=2048)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--compile-mode", default="max-autotune")
    parser.add_argument("--b1-timeout", type=int, default=900)
    parser.add_argument("--streamed-timeout", type=int, default=3600)
    parser.add_argument("--native-timeout", type=int, default=1800)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "runs" / "benchmarks" / "timeline",
    )
    parser.add_argument("--run-id")
    parser.add_argument("--environment-id", default="unlocked")
    parser.add_argument("--source-revision")
    parser.add_argument("--allow-busy-gpu", action="store_true")
    return parser.parse_args(argv)


def selected_checkpoints(value: str) -> list[str]:
    if value == "primary":
        return list(PRIMARY_CHECKPOINTS)
    ids = [item.strip() for item in value.split(",") if item.strip()]
    if not ids:
        raise ValueError("--checkpoints cannot be empty")
    for checkpoint_id in ids:
        resolve_checkpoint(checkpoint_id)
    return ids


def worker_command(args: argparse.Namespace, checkpoint_id: str, stage: str) -> list[str]:
    return [
        args.python,
        "-m",
        "tools.shape14.checkpoint_worker",
        "--checkpoint",
        checkpoint_id,
        "--stage",
        stage,
        "--device",
        args.device,
        "--seed",
        str(args.seed),
        "--batch-limit",
        str(args.batch_limit),
        "--query-chunk",
        str(args.query_chunk),
        "--compare-token-chunk",
        str(args.compare_token_chunk),
        "--warmup",
        str(args.warmup),
        "--repeats",
        str(args.repeats),
        "--compile-mode",
        args.compile_mode,
    ]


def stage_timeout(args: argparse.Namespace, stage: str) -> int:
    if stage == "b1-accuracy":
        return args.b1_timeout
    if stage == "streamed-b32-accuracy":
        return args.streamed_timeout
    return args.native_timeout


def parse_stage_marker(output: str) -> dict | None:
    for line in reversed(output.splitlines()):
        if line.startswith(RESULT_MARKER):
            try:
                return json.loads(line[len(RESULT_MARKER) :])
            except json.JSONDecodeError:
                return None
    return None


def run_stage(args: argparse.Namespace, checkpoint_id: str, stage: str) -> dict:
    command = worker_command(args, checkpoint_id, stage)
    started = time.monotonic()
    try:
        process = subprocess.run(
            command,
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=stage_timeout(args, stage),
        )
        output, returncode = process.stdout, process.returncode
        result = parse_stage_marker(output)
        if result is None:
            lowered = output.lower()
            status = "OOM" if "out of memory" in lowered else "ERROR"
            result = {
                "checkpoint_id": checkpoint_id,
                "stage": stage,
                "status": status,
                "error": "worker result marker missing",
            }
    except subprocess.TimeoutExpired as exc:
        raw = exc.stdout or ""
        output = raw.decode() if isinstance(raw, bytes) else raw
        returncode = None
        result = {
            "checkpoint_id": checkpoint_id,
            "stage": stage,
            "status": "TIMEOUT",
            "error": f"stage exceeded {stage_timeout(args, stage)} seconds",
        }
    except OSError as exc:
        output, returncode = str(exc), None
        result = {
            "checkpoint_id": checkpoint_id,
            "stage": stage,
            "status": "ERROR",
            "error": str(exc),
        }

    result.update(
        {
            "command": shlex.join(command),
            "timeout_s": stage_timeout(args, stage),
            "duration_s": round(time.monotonic() - started, 3),
            "returncode": returncode,
            "output": output,
        }
    )
    return result


def preflight(args: argparse.Namespace, checkpoint_id: str) -> dict:
    command = [
        args.python,
        "-m",
        "tools.timeline_adapter",
        "--checkpoint",
        checkpoint_id,
        "--preflight-only",
    ]
    process = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=60,
    )
    return {
        "status": "PASS" if process.returncode == 0 else "ERROR",
        "returncode": process.returncode,
        "command": shlex.join(command),
        "output": process.stdout,
    }


def skipped_stage(checkpoint_id: str, stage: str, status: str, reason: str) -> dict:
    return {
        "checkpoint_id": checkpoint_id,
        "stage": stage,
        "status": status,
        "reason": reason,
        "command": None,
        "returncode": None,
        "output": "",
    }


def checkpoint_row(record: dict) -> dict:
    stages = record["stages"]
    streamed = stages["streamed-b32-accuracy"]
    timing = stages["native-timing"]
    return {
        "checkpoint_id": record["checkpoint_id"],
        "checkpoint_label": record["checkpoint"]["label"],
        "strategy": record["checkpoint"]["shape14_strategy"],
        "b1_strict": stages["b1-accuracy"]["status"],
        "streamed_b32_strict": streamed["status"],
        "native_b32": stages["native-b32-probe"]["status"],
        "timing": timing["status"],
        "failed": streamed.get("failed"),
        "total": streamed.get("total"),
        "max_abs": streamed.get("max_abs"),
        "max_rel": streamed.get("max_rel"),
        "optimized_median_ms": timing.get("optimized_median_ms"),
        "optimized_mean_ms": timing.get("optimized_mean_ms"),
        "optimized_p90_ms": timing.get("optimized_p90_ms"),
        "throughput_tokens_per_s": timing.get("throughput_tokens_per_s"),
        "peak_allocated_gib": timing.get(
            "peak_allocated_gib",
            stages["native-b32-probe"].get("peak_allocated_gib"),
        ),
        "reason": stages["b1-accuracy"].get("reason"),
    }


def save(run_dir: Path, metadata: dict, records: list[dict]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": {
            **metadata,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        },
        "records": records,
    }
    (run_dir / "shape14_matrix.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    with (run_dir / "shape14_matrix.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(checkpoint_row(record) for record in records)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.list_checkpoints:
        for checkpoint_id, spec in CHECKPOINTS.items():
            print(f"{checkpoint_id:<16} {spec.shape14_strategy:<19} {spec.label}")
        return 0
    try:
        checkpoint_ids = selected_checkpoints(args.checkpoints)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if args.batch_limit != 32:
        raise SystemExit("official shape-#14 sweep requires --batch-limit 32")
    if min(
        args.query_chunk,
        args.compare_token_chunk,
        args.repeats,
        args.b1_timeout,
        args.streamed_timeout,
        args.native_timeout,
    ) <= 0 or args.warmup < 0:
        raise SystemExit("chunk/repeat/timeout values must be positive; warmup non-negative")

    actual_revision = git_revision()
    if (
        args.source_revision
        and actual_revision is not None
        and actual_revision != args.source_revision
    ):
        raise SystemExit(
            f"source revision mismatch: expected {args.source_revision}, got {actual_revision}"
        )
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = args.output_dir / run_id / "shape14"
    metadata = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "environment_id": args.environment_id,
        "source_revision": args.source_revision or actual_revision,
        "source_revision_verification": (
            "git-head"
            if actual_revision is not None
            else "declared-snapshot-with-file-sha256"
        ),
        "platform": platform.platform(),
        "python_executable": args.python,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "runtime": runtime_inventory(args.python),
        "gpu_at_start": gpu_inventory(),
        "runner_sha256": sha256_file(Path(__file__)),
        "worker_sha256": sha256_file(WORKER),
        "adapter_sha256": sha256_file(ADAPTER),
        "checkpoint_ids": checkpoint_ids,
        "protocol": {
            "seed": args.seed,
            "batch_limit": args.batch_limit,
            "query_chunk": args.query_chunk,
            "compare_token_chunk": args.compare_token_chunk,
            "warmup": args.warmup,
            "repeats": args.repeats,
            "compile_mode": args.compile_mode,
            "b1_timeout": args.b1_timeout,
            "streamed_timeout": args.streamed_timeout,
            "native_timeout": args.native_timeout,
            "baseline_latency": None,
            "speedup": None,
        },
        "invocation": shlex.join([sys.executable, *sys.argv]),
    }

    records: list[dict] = []
    overall_ok = True
    for index, checkpoint_id in enumerate(checkpoint_ids, 1):
        spec = resolve_checkpoint(checkpoint_id)
        print(f"[{index}/{len(checkpoint_ids)}] {checkpoint_id}: {spec.label}", flush=True)
        checkpoint_preflight = preflight(args, checkpoint_id)
        record = {
            "checkpoint_id": checkpoint_id,
            "checkpoint": checkpoint_manifest(spec),
            "preflight": checkpoint_preflight,
            "gpu_before": gpu_inventory(),
            "stages": {},
        }
        records.append(record)
        if checkpoint_preflight["status"] != "PASS":
            overall_ok = False
            for stage in STAGE_ORDER:
                record["stages"][stage] = skipped_stage(
                    checkpoint_id, stage, "SKIP", "checkpoint preflight failed"
                )
            save(run_dir, metadata, records)
            continue
        if not args.allow_busy_gpu and not gpu_is_idle(record["gpu_before"]):
            overall_ok = False
            for stage in STAGE_ORDER:
                record["stages"][stage] = skipped_stage(
                    checkpoint_id, stage, "SKIP", "GPU idle gate failed"
                )
            save(run_dir, metadata, records)
            continue
        if spec.shape14_strategy == "static_infeasible":
            reason = spec.shape14_static_reason or "static feasibility gate"
            record["stages"]["b1-accuracy"] = skipped_stage(
                checkpoint_id, "b1-accuracy", "INFEASIBLE_STATIC", reason
            )
            for stage in STAGE_ORDER[1:]:
                record["stages"][stage] = skipped_stage(
                    checkpoint_id, stage, "SKIP", reason
                )
            save(run_dir, metadata, records)
            print(f"    INFEASIBLE_STATIC: {reason}", flush=True)
            continue

        b1 = run_stage(args, checkpoint_id, "b1-accuracy")
        record["stages"]["b1-accuracy"] = b1
        print(f"    b1-accuracy: {b1['status']}", flush=True)
        if b1["status"] != "PASS":
            overall_ok = False
            reason = f"B1 gate was {b1['status']}"
            for stage in STAGE_ORDER[1:]:
                record["stages"][stage] = skipped_stage(
                    checkpoint_id, stage, "SKIP", reason
                )
            save(run_dir, metadata, records)
            continue

        streamed = run_stage(args, checkpoint_id, "streamed-b32-accuracy")
        record["stages"]["streamed-b32-accuracy"] = streamed
        print(f"    streamed-b32-accuracy: {streamed['status']}", flush=True)
        if streamed["status"] != "PASS":
            overall_ok = False
            reason = f"streamed strict gate was {streamed['status']}"
            for stage in STAGE_ORDER[2:]:
                record["stages"][stage] = skipped_stage(
                    checkpoint_id, stage, "SKIP", reason
                )
            save(run_dir, metadata, records)
            continue

        native = run_stage(args, checkpoint_id, "native-b32-probe")
        record["stages"]["native-b32-probe"] = native
        print(f"    native-b32-probe: {native['status']}", flush=True)
        if native["status"] != "PASS":
            record["stages"]["native-timing"] = skipped_stage(
                checkpoint_id,
                "native-timing",
                "N/A",
                f"native B32 probe was {native['status']}",
            )
            save(run_dir, metadata, records)
            continue

        timing = run_stage(args, checkpoint_id, "native-timing")
        record["stages"]["native-timing"] = timing
        print(f"    native-timing: {timing['status']}", flush=True)
        if timing["status"] != "PASS":
            overall_ok = False
        save(run_dir, metadata, records)

    metadata["finished_at"] = datetime.now(timezone.utc).isoformat()
    save(run_dir, metadata, records)
    print(f"Artifacts: {run_dir}")
    return 0 if overall_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
