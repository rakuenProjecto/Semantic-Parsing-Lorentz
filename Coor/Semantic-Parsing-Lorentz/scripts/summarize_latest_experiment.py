#!/usr/bin/env python
"""Summarize the latest experiment diagnostics and training log."""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional


PROJECT_DIR = Path(__file__).resolve().parents[1]


FINAL_RE = re.compile(r"epoch=(?P<epoch>\d+) train_loss=.* val_loss=(?P<val_loss>[-+0-9.eE]+) val_acc=(?P<val_acc>[-+0-9.eE]+)")
BEST_RE = re.compile(r"training complete best_val_loss=(?P<best>[-+0-9.eE]+)")
JAC_RE = re.compile(r"jac_reg=(?P<jac_reg>[-+0-9.eE]+).*jac_complexity=(?P<jac_complexity>[-+0-9.eE]+)")
GPU_RE = re.compile(r"max_allocated_mb=(?P<peak>[-+0-9.eE]+)")
EPOCH_DIAGNOSTICS_RE = re.compile(r"diagnostics_epoch_(?P<epoch>\d+)\.json$")


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def find_latest(pattern: str, root: Path) -> Optional[Path]:
    matches = [path for path in root.glob(pattern) if path.is_file()]
    if not matches:
        return None
    return max(matches, key=lambda path: path.stat().st_mtime)


def resolve_project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_DIR / path


def find_latest_epoch_diagnostics(root: Path) -> Optional[Path]:
    matches = []
    for path in root.glob("diagnostics_epoch_*.json"):
        match = EPOCH_DIAGNOSTICS_RE.fullmatch(path.name)
        if match and path.is_file():
            matches.append((int(match.group("epoch")), path.stat().st_mtime, path))
    if not matches:
        return None
    return max(matches, key=lambda item: (item[0], item[1]))[2]


def git_commit() -> Optional[str]:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_DIR, text=True).strip()
    except Exception:
        return None


def parse_log(path: Optional[Path]) -> Dict[str, Any]:
    parsed: Dict[str, Any] = {
        "best_val_loss": None,
        "final_val_loss": None,
        "final_val_acc": None,
        "jac_reg": None,
        "jac_complexity": None,
        "gpu_mem_peak_mb": None,
        "effective_batch_size": None,
        "amp_enabled": None,
    }
    if path is None or not path.exists():
        return parsed
    text = path.read_text(encoding="utf-8", errors="replace")
    for match in FINAL_RE.finditer(text):
        parsed["final_val_loss"] = float(match.group("val_loss"))
        parsed["final_val_acc"] = float(match.group("val_acc"))
    for match in BEST_RE.finditer(text):
        parsed["best_val_loss"] = float(match.group("best"))
    for match in JAC_RE.finditer(text):
        parsed["jac_reg"] = float(match.group("jac_reg"))
        parsed["jac_complexity"] = float(match.group("jac_complexity"))
    for match in GPU_RE.finditer(text):
        parsed["gpu_mem_peak_mb"] = float(match.group("peak"))
    eff = re.findall(r"effective_batch_size=(\d+)", text)
    amp = re.findall(r"amp_enabled=(true|false)", text)
    if eff:
        parsed["effective_batch_size"] = int(eff[-1])
    if amp:
        parsed["amp_enabled"] = amp[-1] == "true"
    return parsed


def summarize(output_dir: Optional[Path], log_path: Optional[Path], cycle_id: Optional[str]) -> Dict[str, Any]:
    diagnostics_path: Optional[Path] = None
    diagnostics_display_path: Optional[Path] = None
    output_dir_fs: Optional[Path] = None
    output_dir_display = output_dir
    if output_dir is not None:
        output_dir_fs = resolve_project_path(output_dir)
        if not output_dir_fs.is_dir():
            raise FileNotFoundError(f"--output-dir does not exist or is not a directory: {output_dir}")
        diagnostics_path = find_latest_epoch_diagnostics(output_dir_fs)
        if diagnostics_path is None:
            final_path = output_dir_fs / "diagnostics_final.json"
            diagnostics_path = final_path if final_path.exists() else None
        if diagnostics_path is None:
            raise FileNotFoundError(f"no diagnostics_epoch_*.json or diagnostics_final.json found in --output-dir: {output_dir}")
        diagnostics_display_path = output_dir / diagnostics_path.name
    else:
        diagnostics_path = find_latest("**/diagnostics_final.json", PROJECT_DIR / "outputs")
        diagnostics_display_path = diagnostics_path
        output_dir_fs = diagnostics_path.parent if diagnostics_path is not None else None
        output_dir_display = output_dir_fs
    diagnostics = load_json(diagnostics_path) if diagnostics_path is not None else {}
    if log_path is None:
        log_path = find_latest("*.log", PROJECT_DIR / "logs")
    log_metrics = parse_log(resolve_project_path(log_path) if log_path is not None else None)
    gpu_probe_path = PROJECT_DIR / "reports" / "gpu_probe.json"
    gpu = load_json(gpu_probe_path) if gpu_probe_path.exists() else {}
    config_path = output_dir_fs / "config_used.yaml" if output_dir_fs is not None else None
    batch_size = None
    if config_path is not None and config_path.exists():
        for line in config_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("batch_size:"):
                batch_size = int(float(line.split(":", 1)[1].strip()))
                break
    payload = {
        "output_dir": str(output_dir_display) if output_dir_display else None,
        "log_path": str(log_path) if log_path else None,
        "diagnostics_path": str(diagnostics_display_path) if diagnostics_display_path else None,
        "best_val_loss": log_metrics["best_val_loss"],
        "final_val_loss": log_metrics["final_val_loss"],
        "final_val_acc": log_metrics["final_val_acc"],
        "pred_abs_c_mean": diagnostics.get("pred_abs_c_mean", diagnostics.get("mean_abs_c")),
        "pred_abs_c_std": diagnostics.get("pred_abs_c_std", diagnostics.get("std_abs_c")),
        "pred_abs_c_min": diagnostics.get("pred_abs_c_min"),
        "pred_abs_c_max": diagnostics.get("pred_abs_c_max"),
        "target_abs_c_mean": diagnostics.get("target_abs_c_mean", diagnostics.get("target_abs_curvature_mean")),
        "target_abs_c_std": diagnostics.get("target_abs_c_std"),
        "curvature_min_fraction": diagnostics.get("curvature_min_fraction"),
        "curvature_max_fraction": diagnostics.get("curvature_max_fraction"),
        "curvature_mean_to_min_ratio": diagnostics.get("curvature_mean_to_min_ratio"),
        "curvature_collapse_penalty": diagnostics.get("curvature_collapse_penalty"),
        "curvature_spread_penalty": diagnostics.get("curvature_spread_penalty"),
        "curvature_total_aux_loss": diagnostics.get("curvature_total_aux_loss"),
        "jac_frob_mean": diagnostics.get("jac_frob_mean"),
        "jac_frob_std": diagnostics.get("jac_frob_std"),
        "jac_reg": log_metrics["jac_reg"],
        "jac_complexity": log_metrics["jac_complexity"],
        "corr_complexity_abs_c_val": diagnostics.get("corr_complexity_abs_c"),
        "corr_complexity_abs_c_val_valid": diagnostics.get("corr_complexity_abs_c_valid"),
        "corr_complexity_jac_val": diagnostics.get("corr_complexity_jac"),
        "corr_complexity_jac_val_valid": diagnostics.get("corr_complexity_jac_valid"),
        "gpu_name": gpu.get("gpu_name"),
        "gpu_mem_peak_mb": log_metrics["gpu_mem_peak_mb"],
        "batch_size": batch_size,
        "effective_batch_size": log_metrics["effective_batch_size"],
        "amp_enabled": log_metrics["amp_enabled"],
        "cycle_id": cycle_id,
        "git_commit": git_commit(),
    }
    return payload


def write_reports(payload: Dict[str, Any], reports_dir: Path, cycle_id: Optional[str], timestamp: Optional[str]) -> None:
    reports_dir.mkdir(parents=True, exist_ok=True)
    history_dir = reports_dir / "history"
    history_dir.mkdir(parents=True, exist_ok=True)
    json_text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    (reports_dir / "latest_experiment_summary.json").write_text(json_text, encoding="utf-8")
    md = ["# Latest Experiment Summary", ""]
    for key, value in payload.items():
        if isinstance(value, float) and not math.isfinite(value):
            value = None
        md.append(f"- {key}: `{value}`")
    md_text = "\n".join(md) + "\n"
    (reports_dir / "latest_experiment_summary.md").write_text(md_text, encoding="utf-8")
    if cycle_id and timestamp:
        stem = f"summary_cycle_{cycle_id}_{timestamp}"
        (history_dir / f"{stem}.json").write_text(json_text, encoding="utf-8")
        (history_dir / f"{stem}.md").write_text(md_text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir")
    parser.add_argument("--log")
    parser.add_argument("--cycle-id")
    parser.add_argument("--timestamp")
    parser.add_argument("--reports-dir", default=str(PROJECT_DIR / "reports"))
    args = parser.parse_args()
    payload = summarize(
        Path(args.output_dir) if args.output_dir else None,
        Path(args.log) if args.log else None,
        args.cycle_id,
    )
    write_reports(payload, Path(args.reports_dir), args.cycle_id, args.timestamp)
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
