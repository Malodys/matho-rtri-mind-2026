# -*- coding: utf-8 -*-
"""Write a compact, source-backed release manifest for the submission build."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "processed" / "release_manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(*args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(ROOT), *args], capture_output=True, text=True,
        encoding="utf-8", errors="replace", check=False,
    )
    return result.stdout.strip()


def package_versions(names: list[str]) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def file_hashes(paths: list[Path]) -> dict[str, dict[str, int | str]]:
    result: dict[str, dict[str, int | str]] = {}
    for path in paths:
        if path.is_file():
            result[path.relative_to(ROOT).as_posix()] = {
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
    return result


def external_sources() -> list[dict[str, str]]:
    path = ROOT / "external" / "source_manifest.csv"
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def main() -> int:
    parser = argparse.ArgumentParser(description="生成提交发布清单")
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--test-status", choices=("passed", "failed", "not_run"), default="not_run")
    parser.add_argument("--test-summary", default="", help="例如：pytest -q: 90 passed")
    args = parser.parse_args()

    raw_files = sorted((ROOT / "data" / "raw").rglob("*")) if (ROOT / "data" / "raw").is_dir() else []
    artifact_paths = [
        ROOT / "requirements.lock.txt",
        ROOT / "README.md",
        ROOT / "external" / "source_manifest.csv",
        ROOT / "data" / "interim" / "chem_log10_matrix_v1.csv",
        ROOT / "data" / "interim" / "label_dictionary_v1.csv",
        ROOT / "data" / "splits" / "seed_0" / "split_manifest.csv",
        ROOT / "data" / "processed" / "q1_final" / "model_summary.csv",
        ROOT / "data" / "processed" / "q1_final" / "model_independent_gbr.joblib",
        ROOT / "data" / "processed" / "q1_final" / "model_independent_gbr_metadata.json",
        ROOT / "data" / "processed" / "q1_final" / "absolute_grade_metrics_by_label.csv",
        ROOT / "data" / "processed" / "q2_runs" / "summary.json",
        ROOT / "data" / "processed" / "q3_runs" / "generation_summary.csv",
        ROOT / "data" / "processed" / "q4_consumer" / "price_regression.csv",
        ROOT / "data" / "processed" / "paper" / "paper.docx",
        ROOT / "data" / "processed" / "paper" / "paper.pdf",
        ROOT / "data" / "processed" / "paper" / "paper_check.json",
        ROOT / "data" / "processed" / "paper" / "paper_check.txt",
    ]
    head = git("rev-parse", "HEAD")
    status = git("status", "--short")
    model_metadata_path = ROOT / "data" / "processed" / "q1_final" / "model_independent_gbr_metadata.json"
    model_metadata = {}
    if model_metadata_path.is_file():
        model_metadata = json.loads(model_metadata_path.read_text(encoding="utf-8"))
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git": {
            "head": head,
            "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
            "status_clean": not bool(status),
            "status_short": status,
        },
        "python": {
            "executable": str(Path(sys.executable).resolve()),
            "version": platform.python_version(),
            "platform": platform.platform(),
        },
        "dependencies": package_versions([
            "numpy", "pandas", "scikit-learn", "scipy", "torch", "tabpfn",
            "lightgbm", "shap", "copulas", "python-docx", "pypdf", "pdfplumber",
        ]),
        "tests": {
            "status": args.test_status,
            "summary": args.test_summary,
            "command": "python -m pytest -q",
        },
        "git_diff_check": {
            "passed": subprocess.run(
                ["git", "-C", str(ROOT), "diff", "--check"],
                capture_output=True, check=False,
            ).returncode == 0,
        },
        "data_hashes": file_hashes(raw_files),
        "artifact_hashes": file_hashes(artifact_paths),
        "model_contract": {
            "artifact": "data/processed/q1_final/model_independent_gbr.joblib",
            "metadata": "data/processed/q1_final/model_independent_gbr_metadata.json",
            "n_features": len(model_metadata.get("feature_columns", [])),
            "n_targets": len(model_metadata.get("target_columns", [])),
            "n_training_rows": model_metadata.get("n_train_rows", 250),
            "holdout_protocol": "fixed style-stratified 175/75 split",
        },
        "external_sources": external_sources(),
        "external_data_policy": "validation_and_metadata_only; no external rows in main training set",
    }
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": output.relative_to(ROOT).as_posix(), "git": head,
                      "status_clean": payload["git"]["status_clean"],
                      "artifact_count": len(payload["artifact_hashes"])}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
