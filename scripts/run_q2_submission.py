#!/usr/bin/env python3
"""Single entry point for the canonical Q2 submission pipeline."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


def run(command: list[str], root: Path) -> None:
    print(" ".join(command), flush=True)
    subprocess.run(command, cwd=root, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--reuse-results", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    config_path = args.config or root / "configs/q2_submission_v1.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output = root / config["output"]

    if not args.reuse_results:
        run([
            sys.executable,
            "scripts/q2_baseline.py",
            "--output", str(output),
            "--bootstrap", str(config["bootstrap"]),
            "--threshold", str(config["network_threshold"]),
            "--feature-scope", config["feature_scope"],
            "--correlation-source", config["correlation_source"],
            "--penalty", str(config["redundancy_penalty"]),
            "--comparison", config["comparison"],
            "--seed", str(config["seed"]),
        ], root)
    run([sys.executable, "scripts/q2_submission_report.py", "--result-dir", str(output)], root)
    run([sys.executable, "scripts/q2_submission_figures.py", "--result-dir", str(output)], root)
    (output / "submission_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    shutil.copyfile(root / "docs/q2_submission_brief.md", output / "q2_final_report.md")
    print(output)


if __name__ == "__main__":
    main()
