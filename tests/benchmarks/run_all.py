"""Run all benchmarks and produce a consolidated JSON report.

Usage:
    python -m tests.benchmarks.run_all [--report-dir /tmp/evergreen_benchmarks]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def run_benchmark(name: str, module: str, report_dir: str) -> dict | None:
    """Run a single benchmark module via pytest, return results if available."""
    env = os.environ.copy()
    env["BENCH_REPORT_DIR"] = report_dir

    print(f"\n{'='*60}")
    print(f"Running: {name}")
    print(f"{'='*60}")

    result = subprocess.run(
        [
            sys.executable, "-m", "pytest",
            module,
            "-v", "-s",
            "--tb=short",
            "-q",
        ],
        env=env,
        cwd=str(Path(__file__).resolve().parents[2]),
    )

    # Read the JSON report if the benchmark produced one
    report_file = os.path.join(report_dir, f"{name}.json")
    if os.path.exists(report_file):
        with open(report_file) as f:
            return json.load(f)

    return {"status": "skipped" if result.returncode == 0 else "failed"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all evergreen-rag benchmarks")
    parser.add_argument(
        "--report-dir",
        default="/tmp/evergreen_benchmarks",
        help="Directory for benchmark reports",
    )
    args = parser.parse_args()

    report_dir = args.report_dir
    os.makedirs(report_dir, exist_ok=True)

    benchmarks = [
        ("embedding", "tests/benchmarks/bench_embedding.py"),
        ("ingest", "tests/benchmarks/bench_ingest.py"),
        ("search", "tests/benchmarks/bench_search.py"),
    ]

    consolidated = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "benchmarks": {},
    }

    for name, module in benchmarks:
        result = run_benchmark(name, module, report_dir)
        consolidated["benchmarks"][name] = result

    # Write consolidated report
    consolidated_path = os.path.join(report_dir, "consolidated.json")
    with open(consolidated_path, "w") as f:
        json.dump(consolidated, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Consolidated report: {consolidated_path}")
    print(f"{'='*60}")
    print(json.dumps(consolidated, indent=2))


if __name__ == "__main__":
    main()
