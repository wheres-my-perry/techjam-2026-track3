#!/usr/bin/env python3
"""Check that a CUDA host is ready before running the official benchmark."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.submission_preflight import verify_submission  # noqa: E402


def _command(*args: str) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            list(args),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
    except OSError as error:
        return {"command": list(args), "returncode": None, "output": str(error)}
    return {
        "command": list(args),
        "returncode": completed.returncode,
        "output": completed.stdout.strip(),
    }


def _other_compute_processes() -> tuple[list[str], dict[str, Any]]:
    probe = _command(
        "nvidia-smi",
        "--query-compute-apps=pid,process_name,used_memory",
        "--format=csv,noheader,nounits",
    )
    rows: list[str] = []
    if probe["returncode"] == 0:
        own_pid = str(os.getpid())
        for row in probe["output"].splitlines():
            pid = row.split(",", 1)[0].strip()
            if row.strip() and pid != own_pid:
                rows.append(row.strip())
    return rows, probe


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--strict-final-environment", action="store_true")
    parser.add_argument("--require-idle", action="store_true")
    parser.add_argument("--skip-smoke", action="store_true")
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()

    errors: list[str] = []
    warnings: list[str] = []
    report: dict[str, Any] = {
        "device": args.device,
        "strict_final_environment": args.strict_final_environment,
    }

    submission = verify_submission(ROOT)
    report["submission"] = submission
    if submission["status"] != "PASS":
        errors.extend(f"submission: {item}" for item in submission["errors"])

    expected = json.loads(
        (ROOT / "results/final/environment.json").read_text(encoding="utf-8")
    )
    expected_software = expected["software"]
    expected_gpu = expected["gpu"]

    idle_rows, process_probe = _other_compute_processes()
    report["compute_process_probe"] = process_probe
    report["other_compute_processes"] = idle_rows
    if idle_rows:
        message = f"other GPU compute processes detected: {idle_rows}"
        if args.require_idle:
            errors.append(message)
        else:
            warnings.append(message)

    inventory = _command(
        "nvidia-smi",
        "--query-gpu=index,name,driver_version,memory.total,pstate,temperature.gpu,power.draw",
        "--format=csv,noheader,nounits",
    )
    report["nvidia_smi"] = inventory
    if inventory["returncode"] != 0:
        errors.append(f"nvidia-smi inventory failed: {inventory['output']}")

    try:
        import torch
    except ImportError as error:
        errors.append(f"PyTorch import failed: {error}")
        torch = None

    if torch is not None:
        report["software"] = {
            "python": sys.version.split()[0],
            "pytorch": torch.__version__,
            "pytorch_cuda": torch.version.cuda,
        }
        try:
            import triton

            report["software"]["triton"] = triton.__version__
        except ImportError:
            report["software"]["triton"] = None

        if not torch.cuda.is_available():
            errors.append("torch.cuda.is_available() is False")
        else:
            device = torch.device(args.device)
            if device.type != "cuda":
                errors.append(f"--device must resolve to CUDA, got {device}")
            else:
                index = device.index if device.index is not None else torch.cuda.current_device()
                properties = torch.cuda.get_device_properties(index)
                report["gpu"] = {
                    "visible_count": torch.cuda.device_count(),
                    "index": index,
                    "name": properties.name,
                    "compute_capability": f"{properties.major}.{properties.minor}",
                    "total_memory_bytes": properties.total_memory,
                }
                report["features"] = {
                    "torch_compile": hasattr(torch, "compile"),
                    "flash_sdp_enabled": torch.backends.cuda.flash_sdp_enabled(),
                    "flash_attention_available": (
                        torch.backends.cuda.is_flash_attention_available()
                        if hasattr(torch.backends.cuda, "is_flash_attention_available")
                        else None
                    ),
                    "cudnn_sdp_enabled": (
                        torch.backends.cuda.cudnn_sdp_enabled()
                        if hasattr(torch.backends.cuda, "cudnn_sdp_enabled")
                        else None
                    ),
                }
                if not report["features"]["torch_compile"]:
                    errors.append("torch.compile is unavailable")
                if not report["features"]["flash_sdp_enabled"]:
                    errors.append("Flash SDPA is disabled")

                if args.strict_final_environment:
                    expected_versions = {
                        "pytorch": expected_software["pytorch"],
                        "pytorch_cuda": expected_software["pytorch_cuda"],
                        "triton": expected_software["triton"],
                    }
                    for key, expected_value in expected_versions.items():
                        actual = report["software"].get(key)
                        if actual != expected_value:
                            errors.append(
                                f"{key} mismatch: expected {expected_value}, got {actual}"
                            )
                    if sys.version_info[:2] != (3, 12):
                        errors.append(
                            f"Python mismatch: expected 3.12.x, got {sys.version.split()[0]}"
                        )
                    if properties.name != expected_gpu["model"]:
                        errors.append(
                            f"GPU mismatch: expected {expected_gpu['model']}, got {properties.name}"
                        )
                    capability = f"{properties.major}.{properties.minor}"
                    if capability != expected_gpu["compute_capability"]:
                        errors.append(
                            "compute capability mismatch: expected "
                            f"{expected_gpu['compute_capability']}, got {capability}"
                        )
                    minimum_bytes = int(expected_gpu["visible_memory_mib"] * 1024**2 * 0.99)
                    if properties.total_memory < minimum_bytes:
                        errors.append(
                            "visible GPU memory is below the final environment: "
                            f"{properties.total_memory} < {minimum_bytes} bytes"
                        )

                if not args.skip_smoke and not errors:
                    import torch_transformer_benchmark as bench
                    from v16_1_clean import UserOptimizedTransformer

                    torch.manual_seed(1234)
                    torch.cuda.manual_seed_all(1234)
                    config = bench.TransformerConfig(2, 8, 16, 4, 16, 2, True)
                    baseline = bench.BaselineTransformer(config).to(device).eval()
                    optimized = UserOptimizedTransformer(config).to(device).eval()
                    bench.copy_model_weights(baseline, optimized, strict=True)
                    valid_mask = torch.tensor(
                        [
                            [True, True, True, True, True, False, False, False],
                            [True, True, True, False, False, False, False, False],
                        ],
                        device=device,
                    )
                    x = torch.randn(2, 8, 16, device=device)
                    x = x.masked_fill(~valid_mask[..., None], 0)
                    with torch.inference_mode():
                        reference = baseline(x, valid_mask)
                        candidate = optimized(x, valid_mask)
                    accuracy = bench.compare_outputs(
                        reference, candidate, rtol=0.02, atol=0.002
                    )
                    report["correctness_smoke"] = {
                        "status": "PASS" if accuracy.passed else "FAIL",
                        "failed_elements": accuracy.failed_elements,
                        "total_elements": accuracy.total_elements,
                        "max_abs": accuracy.max_abs_error,
                        "max_rel": accuracy.max_relative_error,
                    }
                    if not accuracy.passed:
                        errors.append("strict CUDA correctness smoke failed")

    report["warnings"] = warnings
    report["errors"] = errors
    report["status"] = "PASS" if not errors else "FAIL"

    if args.as_json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"GPU preflight: {report['status']}")
        for warning in warnings:
            print(f"WARNING: {warning}")
        for error in errors:
            print(f"ERROR: {error}")
        if report["status"] == "PASS":
            gpu = report.get("gpu", {})
            software = report.get("software", {})
            print(
                f"device: {gpu.get('name')} sm={gpu.get('compute_capability')} "
                f"torch={software.get('pytorch')} cuda={software.get('pytorch_cuda')} "
                f"triton={software.get('triton')}"
            )
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
