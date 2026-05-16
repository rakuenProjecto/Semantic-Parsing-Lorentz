#!/usr/bin/env python
"""Find a conservative true-Jacobian batch size by short train invocations."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List


PROJECT_DIR = Path(__file__).resolve().parents[1]


def is_oom(text: str) -> bool:
    lowered = text.lower()
    return "cuda out of memory" in lowered or "outofmemoryerror" in lowered


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_markdown(path: Path, payload: Dict[str, Any]) -> None:
    rows = ["# Auto Batch Probe", ""]
    rows.append(f"- Success max batch size: `{payload.get('max_success_batch_size')}`")
    rows.append(f"- Safe batch size: `{payload.get('safe_batch_size')}`")
    rows.append("")
    for result in payload.get("results", []):
        rows.append(f"- batch `{result['batch_size']}`: `{result['status']}`")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def probe_batches(candidates: List[int], config: str, dry_run: bool = False) -> Dict[str, Any]:
    results: List[Dict[str, Any]] = []
    successes: List[int] = []
    for batch_size in candidates:
        output_dir = PROJECT_DIR / "outputs" / f"batch_probe_bs{batch_size}"
        cmd = [
            sys.executable,
            "train.py",
            "--config",
            config,
            "--epochs",
            "1",
            "--num_train_samples",
            str(max(batch_size, 2)),
            "--num_val_samples",
            "2",
            "--batch_size",
            str(batch_size),
            "--output_dir",
            str(output_dir),
            "--log_interval",
            "1",
        ]
        if dry_run:
            results.append({"batch_size": batch_size, "status": "dry_run", "command": cmd})
            successes.append(batch_size)
            continue
        completed = subprocess.run(
            cmd,
            cwd=PROJECT_DIR,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        status = "ok" if completed.returncode == 0 else ("oom" if is_oom(completed.stdout) else "failed")
        results.append(
            {
                "batch_size": batch_size,
                "status": status,
                "returncode": completed.returncode,
                "tail": completed.stdout[-2000:],
            }
        )
        if status == "ok":
            successes.append(batch_size)
        elif status == "oom":
            break
    max_success = max(successes) if successes else None
    safe = max_success if max_success is not None else 2
    if max_success is not None and max_success >= 8:
        safe = max(2, max_success // 2)
    return {
        "candidates": candidates,
        "results": results,
        "max_success_batch_size": max_success,
        "safe_batch_size": safe,
        "dry_run": dry_run,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", default="2,4,8,12,16")
    parser.add_argument("--config", default="configs/bert_true_jacobian_anti_collapse_debug.yaml")
    parser.add_argument("--reports-dir", default=str(PROJECT_DIR / "reports"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    candidates = [int(item) for item in args.candidates.split(",") if item.strip()]
    payload = probe_batches(candidates, args.config, dry_run=args.dry_run)
    reports_dir = Path(args.reports_dir)
    write_json(reports_dir / "auto_batch_probe.json", payload)
    write_markdown(reports_dir / "auto_batch_probe.md", payload)
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
