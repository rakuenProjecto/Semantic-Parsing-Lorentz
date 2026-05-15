"""Parse training logs into CSV rows for quick experiment comparison."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Dict, Iterable, List


KEY_VALUE_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)=([-+A-Za-z0-9_.]+)")

FIELDS = [
    "row_type",
    "epoch",
    "step",
    "loss",
    "ce",
    "acc",
    "curvature_loss",
    "metric_reg",
    "mean_abs_c",
    "mean_complexity",
    "target_abs_c_mean",
    "jac_frob_mean",
    "jac_frob_std",
    "jac_reg",
    "jac_complexity",
    "corr_complexity_abs_c_batch",
    "corr_complexity_abs_c_batch_valid",
    "corr_complexity_jac_batch",
    "corr_complexity_jac_batch_valid",
    "corr_complexity_abs_c_val",
    "corr_complexity_abs_c_val_valid",
    "corr_complexity_jac_val",
    "corr_complexity_jac_val_valid",
    "curvature_min_fraction",
    "curvature_max_fraction",
    "curvature_mean_to_min_ratio",
    "curvature_aux_loss",
    "target_abs_curvature_mean",
    "val_loss",
    "val_acc",
    "train_loss",
    "train_acc",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parse Lorentz training logs to CSV")
    parser.add_argument("--log", required=True, help="Training log path")
    parser.add_argument("--out", help="Output CSV path. Defaults to the log path with .csv suffix.")
    return parser.parse_args()


def parse_line(line: str) -> Dict[str, str]:
    pairs = {key: value for key, value in KEY_VALUE_RE.findall(line)}
    if "epoch" not in pairs:
        return {}
    if " step=" in line and "loss" in pairs:
        pairs["row_type"] = "train_step"
    elif "validation_diagnostics" in line:
        pairs["row_type"] = "validation_diagnostics"
    elif "val_loss" in pairs and "val_acc" in pairs:
        pairs["row_type"] = "epoch_summary"
    else:
        return {}
    return pairs


def parse_log(path: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            row = parse_line(line)
            if row:
                rows.append(row)
    return rows


def write_csv(rows: Iterable[Dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in FIELDS})


def main() -> None:
    args = parse_args()
    log_path = Path(args.log)
    out_path = Path(args.out) if args.out else log_path.with_suffix(".csv")
    rows = parse_log(log_path)
    write_csv(rows, out_path)
    print(f"wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
