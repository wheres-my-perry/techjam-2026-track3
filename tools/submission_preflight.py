#!/usr/bin/env python3
"""Verify the frozen submission source identity and curated evidence.

This preflight intentionally uses only the Python standard library so a reviewer
can verify a clean checkout before installing the CUDA runtime.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "submission-manifest.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path.relative_to(ROOT)} must contain a JSON object")
    return value


def _main_wiring_errors(path: Path) -> list[str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError) as error:
        return [f"cannot parse main.py: {error}"]

    imported = any(
        isinstance(node, ast.ImportFrom)
        and node.module == "v16_1_clean"
        and any(alias.name == "UserOptimizedTransformer" for alias in node.names)
        for node in ast.walk(tree)
    )
    assigned = any(
        isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Attribute)
            and isinstance(target.value, ast.Name)
            and target.value.id == "bench"
            and target.attr == "UserOptimizedTransformer"
            for target in node.targets
        )
        and isinstance(node.value, ast.Name)
        and node.value.id == "UserOptimizedTransformer"
        for node in ast.walk(tree)
    )

    errors: list[str] = []
    if not imported:
        errors.append("main.py does not import UserOptimizedTransformer from v16_1_clean")
    if not assigned:
        errors.append("main.py does not install V16.1 as the benchmark implementation")
    return errors


def _same_number(actual: Any, expected: Any) -> bool:
    if isinstance(expected, float):
        return isinstance(actual, (int, float)) and math.isclose(
            float(actual), expected, rel_tol=1e-12, abs_tol=1e-12
        )
    return actual == expected


def verify_submission(root: Path = ROOT) -> dict[str, Any]:
    manifest_path = root / MANIFEST_PATH.name
    errors: list[str] = []
    checks: list[dict[str, Any]] = []

    try:
        manifest = _load_json(manifest_path)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        return {"status": "FAIL", "errors": [str(error)], "checks": []}

    release = manifest.get("release", {})
    measured_revision = release.get("measured_source_revision")
    environment_id = release.get("environment_id")

    locked_files = manifest.get("locked_files")
    if not isinstance(locked_files, list) or not locked_files:
        errors.append("manifest locked_files must be a non-empty list")
    else:
        for item in locked_files:
            relative = item.get("path") if isinstance(item, dict) else None
            expected = item.get("sha256") if isinstance(item, dict) else None
            if not isinstance(relative, str) or not isinstance(expected, str):
                errors.append("every locked_files entry needs path and sha256 strings")
                continue
            path = root / relative
            if not path.is_file():
                errors.append(f"missing locked file: {relative}")
                continue
            actual = sha256_file(path)
            passed = actual == expected
            checks.append(
                {
                    "kind": "sha256",
                    "path": relative,
                    "status": "PASS" if passed else "FAIL",
                    "expected": expected,
                    "actual": actual,
                }
            )
            if not passed:
                errors.append(f"SHA-256 mismatch: {relative}")

    errors.extend(_main_wiring_errors(root / "main.py"))

    try:
        environment = _load_json(root / "results/final/environment.json")
        if environment.get("source_revision") != measured_revision:
            errors.append("environment source_revision differs from the release manifest")
        if environment.get("environment_id") != environment_id:
            errors.append("environment_id differs from the release manifest")
    except (OSError, ValueError, json.JSONDecodeError) as error:
        errors.append(f"invalid final environment artifact: {error}")

    expectations = manifest.get("evidence_expectations", {})
    shapes_expectation = expectations.get("shapes_1_13", {})
    try:
        shapes = _load_json(root / shapes_expectation["path"])
        summary = shapes["summary"]
        if shapes.get("metadata", {}).get("source_revision") != measured_revision:
            errors.append("shapes 1-13 source_revision differs from the manifest")
        for key in (
            "status",
            "passed_shapes",
            "total_shapes",
            "failed_elements",
            "measured_speedup_geomean",
        ):
            if not _same_number(summary.get(key), shapes_expectation.get(key)):
                errors.append(f"shapes 1-13 evidence mismatch: summary.{key}")
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        errors.append(f"invalid shapes 1-13 evidence: {error}")

    shape14_expectation = expectations.get("shape_14", {})
    try:
        shape14 = _load_json(root / shape14_expectation["path"])
        streamed = shape14["streamed_b32_accuracy"]
        native = shape14["native_b32_probe"]
        performance = shape14["performance"]
        if shape14.get("source_revision") != measured_revision:
            errors.append("shape 14 source_revision differs from the manifest")
        mappings = (
            (streamed.get("status"), shape14_expectation.get("streamed_status"), "streamed status"),
            (streamed.get("failed_elements"), shape14_expectation.get("failed_elements"), "failed elements"),
            (streamed.get("total_elements"), shape14_expectation.get("total_elements"), "total elements"),
            (native.get("status"), shape14_expectation.get("native_status"), "native status"),
            (performance.get("median_ms"), shape14_expectation.get("median_ms"), "median_ms"),
            (
                performance.get("throughput_tokens_per_second"),
                shape14_expectation.get("throughput_tokens_per_second"),
                "throughput",
            ),
            (
                performance.get("peak_allocated_gib"),
                shape14_expectation.get("peak_allocated_gib"),
                "peak allocation",
            ),
            (performance.get("speedup"), shape14_expectation.get("baseline_speedup"), "baseline speedup"),
        )
        for actual, expected, label in mappings:
            if not _same_number(actual, expected):
                errors.append(f"shape 14 evidence mismatch: {label}")
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        errors.append(f"invalid shape 14 evidence: {error}")

    return {
        "status": "PASS" if not errors else "FAIL",
        "release_tag": release.get("tag"),
        "measured_source_revision": measured_revision,
        "locked_file_count": len(locked_files) if isinstance(locked_files, list) else 0,
        "checks": checks,
        "errors": errors,
    }


def _git_output(root: Path, *args: str) -> tuple[int, str]:
    completed = subprocess.run(
        ["git", *args],
        cwd=root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return completed.returncode, completed.stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-clean", action="store_true")
    parser.add_argument("--require-tag", action="store_true")
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()

    result = verify_submission(ROOT)
    errors = result["errors"]

    if args.require_clean:
        returncode, output = _git_output(ROOT, "status", "--porcelain")
        if returncode != 0:
            errors.append(f"git status failed: {output}")
        elif output:
            errors.append("working tree is not clean")

    if args.require_tag:
        expected_tag = result.get("release_tag")
        returncode, output = _git_output(ROOT, "tag", "--points-at", "HEAD")
        tags = output.splitlines() if returncode == 0 else []
        if expected_tag not in tags:
            errors.append(f"HEAD is not tagged {expected_tag}")

    result["status"] = "PASS" if not errors else "FAIL"
    if args.as_json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(
            f"submission preflight: {result['status']} "
            f"({result['locked_file_count']} locked files)"
        )
        if result["status"] == "PASS":
            print(f"release tag: {result['release_tag']}")
            print(f"measured revision: {result['measured_source_revision']}")
        for error in errors:
            print(f"ERROR: {error}")
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
