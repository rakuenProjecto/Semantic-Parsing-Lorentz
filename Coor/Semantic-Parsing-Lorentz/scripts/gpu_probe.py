#!/usr/bin/env python
"""Probe GPU availability and write lightweight reports."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List


PROJECT_DIR = Path(__file__).resolve().parents[1]


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def recommended_batches(free_mb: float) -> List[int]:
    if free_mb >= 18000:
        return [2, 4, 8, 12, 16]
    if free_mb >= 10000:
        return [2, 4, 8]
    if free_mb >= 6000:
        return [2, 4]
    return [2]


def probe_gpu() -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "cuda_available": False,
        "gpu_name": None,
        "total_vram_mb": 0,
        "used_vram_mb": 0,
        "free_vram_mb": 0,
        "recommended_batch_candidates": [2],
    }
    try:
        import torch

        payload["cuda_available"] = bool(torch.cuda.is_available())
    except Exception as exc:
        payload["torch_error"] = str(exc)
        return payload

    if not payload["cuda_available"]:
        return payload

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.used,memory.free",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        line = result.stdout.strip().splitlines()[0]
        name, total, used, free = [part.strip() for part in line.split(",")]
        payload.update(
            {
                "gpu_name": name,
                "total_vram_mb": float(total),
                "used_vram_mb": float(used),
                "free_vram_mb": float(free),
                "recommended_batch_candidates": recommended_batches(float(free)),
            }
        )
    except Exception as exc:
        payload["nvidia_smi_error"] = str(exc)
    return payload


def write_markdown(path: Path, payload: Dict[str, Any]) -> None:
    lines = [
        "# GPU Probe",
        "",
        f"- CUDA available: `{payload.get('cuda_available')}`",
        f"- GPU name: `{payload.get('gpu_name')}`",
        f"- Total VRAM MB: `{payload.get('total_vram_mb')}`",
        f"- Used VRAM MB: `{payload.get('used_vram_mb')}`",
        f"- Free VRAM MB: `{payload.get('free_vram_mb')}`",
        f"- Recommended batch candidates: `{payload.get('recommended_batch_candidates')}`",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports-dir", default=str(PROJECT_DIR / "reports"))
    args = parser.parse_args()
    reports_dir = Path(args.reports_dir)
    payload = probe_gpu()
    write_json(reports_dir / "gpu_probe.json", payload)
    write_markdown(reports_dir / "gpu_probe.md", payload)
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
