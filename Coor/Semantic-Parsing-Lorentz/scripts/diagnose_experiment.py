#!/usr/bin/env python
"""Diagnose experiment summaries for collapse, Jacobian, learning, and GPU issues."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List


PROJECT_DIR = Path(__file__).resolve().parents[1]


def finite_positive(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value)) and float(value) > 0.0


def diagnose(summary: Dict[str, Any]) -> Dict[str, Any]:
    issues: List[Dict[str, str]] = []
    recommendations: List[str] = []

    if (summary.get("curvature_min_fraction") or 0.0) > 0.5:
        issues.append({"type": "curvature_collapse", "severity": "high", "message": "curvature_min_fraction > 0.5"})
    if summary.get("pred_abs_c_mean") is not None and summary["pred_abs_c_mean"] < 0.10:
        issues.append({"type": "floor_sticking", "severity": "high", "message": "pred_abs_c_mean < 0.10"})
    if summary.get("pred_abs_c_std") is not None and summary["pred_abs_c_std"] < 0.005:
        issues.append({"type": "low_curvature_diversity", "severity": "medium", "message": "pred_abs_c_std < 0.005"})
    if summary.get("curvature_mean_to_min_ratio") is not None and summary["curvature_mean_to_min_ratio"] < 2.0:
        issues.append({"type": "near_min_curvature", "severity": "high", "message": "curvature_mean_to_min_ratio < 2.0"})
    if not finite_positive(summary.get("jac_frob_mean")):
        issues.append({"type": "jacobian_invalid", "severity": "high", "message": "jac_frob_mean is missing, non-finite, or zero"})
    if summary.get("final_val_loss") is not None and summary["final_val_loss"] > 5.0:
        issues.append({"type": "learning_failure", "severity": "medium", "message": "final_val_loss is very high"})
    if summary.get("final_val_acc") is not None and summary["final_val_acc"] < 0.15:
        issues.append({"type": "learning_failure", "severity": "medium", "message": "final_val_acc near random"})
    if summary.get("gpu_mem_peak_mb") is not None and summary.get("gpu_mem_peak_mb", 0) < 4000:
        issues.append({"type": "gpu_underuse", "severity": "low", "message": "GPU memory peak appears low"})

    issue_types = {issue["type"] for issue in issues}
    if issue_types & {"curvature_collapse", "floor_sticking", "low_curvature_diversity", "near_min_curvature"}:
        recommendations.extend(
            [
                "increase curvature_anti_collapse_weight",
                "increase curvature_min_margin",
                "increase curvature_spread_weight or curvature_min_std",
                "increase curvature_init_abs_c",
                "avoid target warmup that permits early collapse",
            ]
        )
    if "learning_failure" in issue_types:
        recommendations.extend(
            [
                "lower anti-collapse weights",
                "lower learning_rate",
                "tighten grad_clip_norm",
                "use curvature auxiliary warmup",
            ]
        )
    if "gpu_underuse" in issue_types:
        recommendations.extend(["try larger batch_size", "increase gradient_accumulation_steps", "move short mode toward full"])
    status = "pass" if not issues else "needs_fix"
    return {"status": status, "issues": issues, "recommendations": recommendations, "summary": summary}


def write_reports(payload: Dict[str, Any], reports_dir: Path, cycle_id: str | None, timestamp: str | None) -> None:
    reports_dir.mkdir(parents=True, exist_ok=True)
    history_dir = reports_dir / "history"
    history_dir.mkdir(parents=True, exist_ok=True)
    json_text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    (reports_dir / "latest_diagnosis.json").write_text(json_text, encoding="utf-8")
    lines = ["# Latest Diagnosis", "", f"- status: `{payload['status']}`", ""]
    for issue in payload["issues"]:
        lines.append(f"- {issue['severity']} `{issue['type']}`: {issue['message']}")
    if payload["recommendations"]:
        lines.extend(["", "## Recommendations"])
        lines.extend(f"- {item}" for item in payload["recommendations"])
    md_text = "\n".join(lines) + "\n"
    (reports_dir / "latest_diagnosis.md").write_text(md_text, encoding="utf-8")
    if cycle_id and timestamp:
        stem = f"diagnosis_cycle_{cycle_id}_{timestamp}"
        (history_dir / f"{stem}.json").write_text(json_text, encoding="utf-8")
        (history_dir / f"{stem}.md").write_text(md_text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", default=str(PROJECT_DIR / "reports" / "latest_experiment_summary.json"))
    parser.add_argument("--cycle-id")
    parser.add_argument("--timestamp")
    parser.add_argument("--reports-dir", default=str(PROJECT_DIR / "reports"))
    args = parser.parse_args()
    summary = json.loads(Path(args.summary).read_text(encoding="utf-8"))
    payload = diagnose(summary)
    write_reports(payload, Path(args.reports_dir), args.cycle_id, args.timestamp)
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
