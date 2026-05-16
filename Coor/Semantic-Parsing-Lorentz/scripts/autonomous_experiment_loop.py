#!/usr/bin/env python
"""Autonomous UBAI experiment-analysis-fix loop controller."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


PROJECT_DIR = Path(__file__).resolve().parents[1]
STATE_PATH = PROJECT_DIR / "reports" / "autonomous_loop_state.json"
STOP_PATH = PROJECT_DIR / "STOP_AUTONOMOUS_LOOP"
CONFIG_PATH = PROJECT_DIR / "configs" / "bert_true_jacobian_anti_collapse.yaml"


def as_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return value.lower() in {"1", "true", "yes", "on"}


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def read_json(path: Path, default: Dict[str, Any]) -> Dict[str, Any]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return default


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_command(cmd: List[str], log_path: Path, dry_run: bool = False) -> Dict[str, Any]:
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"\n$ {' '.join(cmd)}\n")
        if dry_run:
            handle.write("[dry-run]\n")
            return {"returncode": 0, "oom": False, "dry_run": True}
        completed = subprocess.run(
            cmd,
            cwd=PROJECT_DIR,
            text=True,
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
    text = log_path.read_text(encoding="utf-8", errors="replace")
    return {
        "returncode": completed.returncode,
        "oom": "cuda out of memory" in text.lower() or "outofmemoryerror" in text.lower(),
        "dry_run": False,
    }


def current_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_DIR, text=True).strip()
    except Exception:
        return None


def load_diagnosis() -> Dict[str, Any]:
    return read_json(PROJECT_DIR / "reports" / "latest_diagnosis.json", {"status": "missing", "issues": []})


def update_config_value(path: Path, key: str, transform) -> str:
    lines = path.read_text(encoding="utf-8").splitlines()
    changed = False
    new_lines: List[str] = []
    old_new = ""
    for line in lines:
        if line.startswith(f"{key}:"):
            old_value = line.split(":", 1)[1].strip()
            try:
                numeric = float(old_value)
                if numeric.is_integer():
                    numeric = int(numeric)
            except ValueError:
                numeric = old_value
            new_value = transform(numeric)
            new_lines.append(f"{key}: {new_value}")
            old_new = f"{key}: {old_value} -> {new_value}"
            changed = True
        else:
            new_lines.append(line)
    if not changed:
        new_value = transform(0)
        new_lines.append(f"{key}: {new_value}")
        old_new = f"{key}: <missing> -> {new_value}"
    path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
    return old_new


def apply_automatic_fix(diagnosis: Dict[str, Any], state: Dict[str, Any]) -> List[str]:
    issues = {issue.get("type") for issue in diagnosis.get("issues", [])}
    changes: List[str] = []
    if issues & {"curvature_collapse", "floor_sticking", "near_min_curvature"}:
        changes.append(update_config_value(CONFIG_PATH, "curvature_anti_collapse_weight", lambda v: min(float(v or 0.5) * 1.5, 3.0)))
        changes.append(update_config_value(CONFIG_PATH, "curvature_min_margin", lambda v: min(float(v or 0.2) + 0.05, 0.6)))
        changes.append(update_config_value(CONFIG_PATH, "curvature_init_abs_c", lambda v: min(float(v or 0.8) + 0.1, 2.0)))
        state["last_fix_type"] = "curvature_collapse"
    elif "low_curvature_diversity" in issues:
        changes.append(update_config_value(CONFIG_PATH, "curvature_spread_weight", lambda v: min(float(v or 0.2) * 1.5, 2.0)))
        changes.append(update_config_value(CONFIG_PATH, "curvature_min_std", lambda v: min(float(v or 0.03) + 0.01, 0.2)))
        state["last_fix_type"] = "curvature_diversity"
    elif "learning_failure" in issues:
        changes.append(update_config_value(CONFIG_PATH, "curvature_anti_collapse_weight", lambda v: max(float(v or 0.5) * 0.7, 0.05)))
        changes.append(update_config_value(CONFIG_PATH, "learning_rate", lambda v: max(float(v or 2e-5) * 0.7, 1e-6)))
        changes.append(update_config_value(CONFIG_PATH, "grad_clip_norm", lambda v: min(float(v or 1.0), 1.0)))
        state["last_fix_type"] = "learning_failure"
    elif "gpu_underuse" in issues:
        state["last_fix_type"] = "gpu_underuse"
        changes.append("gpu_underuse: use auto_batch_probe safe_batch_size in next cycle")
    return changes


def write_next_prompt(state: Dict[str, Any], diagnosis: Dict[str, Any], changes: List[str]) -> None:
    lines = [
        "# Codex Next Action Prompt",
        "",
        f"- cycle: `{state.get('cycle')}`",
        f"- last status: `{state.get('last_status')}`",
        f"- last commit: `{state.get('last_commit_hash')}`",
        "",
        "## Diagnosis",
        json.dumps(diagnosis.get("issues", []), indent=2),
        "",
        "## Changes This Cycle",
        *(f"- {change}" for change in changes),
        "",
        "## Constraints",
        "- Keep true autograd Jacobian path: J_x = dx/dh, C_x = J_x J_x^T, scale = (C_x x)^T W y.",
        "- Do not commit outputs/, logs/, checkpoints, tokens, SSH keys, .env, or private credentials.",
    ]
    (PROJECT_DIR / "reports" / "codex_next_action_prompt.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def commit_and_push(cycle: int, push: bool, dry_run: bool, log_path: Path) -> Dict[str, Any]:
    add_paths = [
        "README.md",
        "configs",
        "scripts",
        "src",
        "tests",
        "train.py",
        "reports/latest_experiment_summary.json",
        "reports/latest_experiment_summary.md",
        "reports/latest_diagnosis.json",
        "reports/latest_diagnosis.md",
        "reports/autonomous_loop_state.json",
        "reports/codex_next_action_prompt.md",
    ]
    for cmd in (["git", "diff", "--check"], ["python", "-m", "pytest", "tests"], ["python", "train.py", "--config", "configs/tiny_true_jacobian_smoke.yaml"]):
        result = run_command(cmd, log_path, dry_run=dry_run)
        if result["returncode"] != 0:
            return {"committed": False, "reason": f"verification failed: {' '.join(cmd)}"}
    run_command(["git", "add", *add_paths], log_path, dry_run=dry_run)
    commit_result = run_command(["git", "commit", "-m", f"Autonomous cycle {cycle}: improve curvature control"], log_path, dry_run=dry_run)
    if commit_result["returncode"] != 0:
        return {"committed": False, "reason": "nothing to commit or commit failed"}
    if push:
        push_result = run_command(["git", "push", "origin", "main"], log_path, dry_run=dry_run)
        if push_result["returncode"] != 0:
            return {"committed": True, "pushed": False, "reason": "push failed"}
    return {"committed": True, "pushed": push and not dry_run}


def plan_for_mode(mode: str) -> List[List[str]]:
    if mode == "auto":
        return [
            ["git", "fetch", "origin"],
            ["git", "status", "--short"],
            ["python", "scripts/gpu_probe.py"],
            ["python", "scripts/auto_batch_probe.py", "--config", "configs/bert_true_jacobian_anti_collapse_debug.yaml"],
            ["bash", "scripts/run_experiment_cycle.sh", "--mode", "short"],
        ]
    return [
        ["git", "fetch", "origin"],
        ["git", "status", "--short"],
        ["bash", "scripts/run_experiment_cycle.sh", "--mode", mode],
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["debug", "short", "full", "auto"], default="auto")
    parser.add_argument("--max-consecutive-failures", type=int, default=3)
    parser.add_argument("--commit-on-success", type=as_bool, default=True)
    parser.add_argument("--push-on-success", type=as_bool, default=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-cycles", type=int, default=0, help="0 means run until stopped")
    args = parser.parse_args()

    (PROJECT_DIR / "logs").mkdir(exist_ok=True)
    (PROJECT_DIR / "reports" / "history").mkdir(parents=True, exist_ok=True)
    state = read_json(
        STATE_PATH,
        {
            "cycle": 0,
            "consecutive_failures": 0,
            "oom_count": 0,
            "improvement_failures": 0,
            "last_status": "new",
            "last_commit_hash": current_commit(),
        },
    )

    while True:
        if STOP_PATH.exists():
            state["last_status"] = "stopped_by_STOP_AUTONOMOUS_LOOP"
            write_json(STATE_PATH, state)
            print("STOP_AUTONOMOUS_LOOP found; exiting")
            return
        if state["consecutive_failures"] >= args.max_consecutive_failures:
            error_path = PROJECT_DIR / "reports" / "autonomous_loop_error.md"
            error_path.write_text("# Autonomous Loop Error\n\nFatal errors reached limit.\n", encoding="utf-8")
            state["last_status"] = "fatal_failure_limit"
            write_json(STATE_PATH, state)
            return
        if args.max_cycles and state["cycle"] >= args.max_cycles:
            state["last_status"] = "max_cycles_reached"
            write_json(STATE_PATH, state)
            return

        state["cycle"] += 1
        cycle = state["cycle"]
        ts = timestamp()
        log_path = PROJECT_DIR / "logs" / f"autonomous_cycle_{cycle}_{ts}.log"
        state.update({"current_state": "running", "last_cycle_timestamp": ts, "last_commit_hash": current_commit()})
        write_json(STATE_PATH, state)

        if args.dry_run:
            state["dry_run_plan"] = plan_for_mode(args.mode)
            state["last_status"] = "dry_run_planned"
            state["current_state"] = "dry_run"
            write_json(STATE_PATH, state)
            write_next_prompt(state, {"issues": []}, ["dry-run only"])
            print(json.dumps(state, indent=2, sort_keys=True))
            return

        cycle_failed = False
        for cmd in plan_for_mode(args.mode):
            result = run_command(cmd, log_path, dry_run=False)
            if result["oom"]:
                state["oom_count"] += 1
            if result["returncode"] != 0:
                state["consecutive_failures"] += 1
                state["last_status"] = f"command_failed: {' '.join(cmd)}"
                cycle_failed = True
                break

        diagnosis = load_diagnosis()
        changes = []
        if not cycle_failed and diagnosis.get("status") != "pass":
            changes = apply_automatic_fix(diagnosis, state)
            run_command(["git", "diff", "--check"], log_path, dry_run=False)
            state["improvement_failures"] = state.get("improvement_failures", 0) + 1
            if state["improvement_failures"] >= 3:
                changes.append("three repeated non-improving fixes: switch hypothesis next cycle")
                state["improvement_failures"] = 0
        elif not cycle_failed:
            state["improvement_failures"] = 0

        write_next_prompt(state, diagnosis, changes)
        if cycle_failed:
            write_json(STATE_PATH, state)
            continue

        state["consecutive_failures"] = 0
        state["last_status"] = "cycle_passed" if diagnosis.get("status") == "pass" else "cycle_fixed_config"
        write_json(STATE_PATH, state)
        if diagnosis.get("status") == "pass" and args.commit_on_success:
            commit_result = commit_and_push(cycle, push=args.push_on_success, dry_run=False, log_path=log_path)
            state["last_commit_result"] = commit_result
            state["last_commit_hash"] = current_commit()
            write_json(STATE_PATH, state)

        time.sleep(5)


if __name__ == "__main__":
    main()
