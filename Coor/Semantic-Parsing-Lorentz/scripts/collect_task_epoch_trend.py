from pathlib import Path
import argparse
import json
import re
from datetime import datetime

EXPERIMENTS = [
    "main",
    "bert_no_jacobian_reg",
    "bert_no_curvature_aux",
    "bert_no_aux_all",
]

METRICS = [
    "train_acc",
    "val_acc",
    "val_loss",
    "curvature_min_fraction",
    "curvature_collapse_penalty",
    "pred_abs_c_mean",
    "corr_complexity_abs_c_val",
    "corr_complexity_jac_val",
    "jac_frob_mean",
]

def read_json(path: Path):
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))

def parse_epoch_val(log_path: Path, epoch: int):
    if not log_path.exists():
        return None, None, None, None

    text = log_path.read_text(encoding="utf-8", errors="replace")
    pattern = re.compile(
        rf"epoch={epoch}\s+train_loss=(?P<train_loss>[-+0-9.eE]+)\s+"
        rf"train_acc=(?P<train_acc>[-+0-9.eE]+)\s+"
        rf"val_loss=(?P<val_loss>[-+0-9.eE]+)\s+"
        rf"val_acc=(?P<val_acc>[-+0-9.eE]+)"
    )
    matches = list(pattern.finditer(text))
    if not matches:
        return None, None, None, None

    m = matches[-1]
    return (
        float(m.group("train_loss")),
        float(m.group("train_acc")),
        float(m.group("val_loss")),
        float(m.group("val_acc")),
    )

def fmt(v):
    if isinstance(v, float):
        return f"{v:.6g}"
    if v is None:
        return ""
    return str(v)

def delta(a, b):
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return b - a
    return None

def make_row(exp, epoch, out_root, log_root):
    output_dir = out_root / exp
    log_path = log_root / f"{exp}.log"
    diag_path = output_dir / f"diagnostics_epoch_{epoch}.json"

    diag = read_json(diag_path)
    train_loss, train_acc, val_loss, val_acc = parse_epoch_val(log_path, epoch)

    return {
        "experiment": exp,
        "epoch": epoch,
        "train_acc": train_acc,
        "val_acc": val_acc,
        "val_loss": val_loss,
        "curvature_min_fraction": diag.get("curvature_min_fraction"),
        "curvature_collapse_penalty": diag.get("curvature_collapse_penalty"),
        "pred_abs_c_mean": diag.get("pred_abs_c_mean", diag.get("mean_abs_c")),
        "corr_complexity_abs_c_val": diag.get("corr_complexity_abs_c"),
        "corr_complexity_jac_val": diag.get("corr_complexity_jac"),
        "jac_frob_mean": diag.get("jac_frob_mean"),
        "output_dir": str(output_dir),
        "log_path": str(log_path),
        "diagnostics_path": str(diag_path),
        "diagnostics_exists": diag_path.exists(),
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="mrpc")
    parser.add_argument("--epoch-a", type=int, default=3)
    parser.add_argument("--epoch-b", type=int, default=7)
    parser.add_argument(
        "--output-root",
        default="outputs_task_isolated/glue_mrpc_e7_a10_isolated_n054.hpc_20260517_213443",
    )
    parser.add_argument("--logs-base", default="logs_task_isolated")
    parser.add_argument("--reports-dir", default="reports/final")
    parser.add_argument("--ubai-reports-dir", default="reports_task_isolated/final")
    args = parser.parse_args()

    task = args.task.lower()
    out_root = Path(args.output_root)
    run_tag = out_root.name
    log_root = Path(args.logs_base) / run_tag

    rows_a = {exp: make_row(exp, args.epoch_a, out_root, log_root) for exp in EXPERIMENTS}
    rows_b = {exp: make_row(exp, args.epoch_b, out_root, log_root) for exp in EXPERIMENTS}

    trend_rows = []
    for exp in EXPERIMENTS:
        ra = rows_a[exp]
        rb = rows_b[exp]
        row = {
            "experiment": exp,
            "epoch_a": args.epoch_a,
            "epoch_b": args.epoch_b,
        }
        for m in METRICS:
            row[f"{m}_e{args.epoch_a}"] = ra.get(m)
            row[f"{m}_e{args.epoch_b}"] = rb.get(m)
            row[f"{m}_delta"] = delta(ra.get(m), rb.get(m))
        trend_rows.append(row)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    reports_dir = Path(args.reports_dir)
    ubai_reports_dir = Path(args.ubai_reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    ubai_reports_dir.mkdir(parents=True, exist_ok=True)

    stem = f"{task}_epoch{args.epoch_a}_to_epoch{args.epoch_b}_trend_{run_tag}_{ts}"
    json_path = reports_dir / f"{stem}.json"
    md_path = reports_dir / f"{stem}.md"
    ubai_json_path = ubai_reports_dir / f"{stem}.json"
    ubai_md_path = ubai_reports_dir / f"{stem}.md"

    payload = {
        "task": task,
        "run_tag": run_tag,
        "output_root": str(out_root),
        "log_root": str(log_root),
        "epoch_a": args.epoch_a,
        "epoch_b": args.epoch_b,
        "rows_epoch_a": list(rows_a.values()),
        "rows_epoch_b": list(rows_b.values()),
        "trend_rows": trend_rows,
    }

    json_text = json.dumps(payload, indent=2, ensure_ascii=False)
    json_path.write_text(json_text, encoding="utf-8")
    ubai_json_path.write_text(json_text, encoding="utf-8")

    lines = []
    lines.append(f"# {task.upper()} Epoch {args.epoch_a} → Epoch {args.epoch_b} Trend Comparison")
    lines.append("")
    lines.append(f"- task: `{task}`")
    lines.append(f"- run_tag: `{run_tag}`")
    lines.append(f"- output_root: `{out_root}`")
    lines.append(f"- log_root: `{log_root}`")
    lines.append("")

    lines.append("## Epoch Final Values")
    lines.append("")
    base_cols = [
        "experiment",
        "epoch",
        "train_acc",
        "val_acc",
        "val_loss",
        "curvature_min_fraction",
        "curvature_collapse_penalty",
        "pred_abs_c_mean",
        "corr_complexity_abs_c_val",
        "corr_complexity_jac_val",
        "jac_frob_mean",
    ]
    lines.append("| " + " | ".join(base_cols) + " |")
    lines.append("|" + "|".join(["---"] + ["---:" for _ in base_cols[1:]]) + "|")
    for exp in EXPERIMENTS:
        for r in [rows_a[exp], rows_b[exp]]:
            lines.append("| " + " | ".join(fmt(r.get(c)) for c in base_cols) + " |")

    lines.append("")
    lines.append("## Delta Table")
    lines.append("")
    delta_cols = [
        "experiment",
        "val_acc_delta",
        "val_loss_delta",
        "curvature_min_fraction_delta",
        "curvature_collapse_penalty_delta",
        "pred_abs_c_mean_delta",
        "corr_complexity_abs_c_val_delta",
        "corr_complexity_jac_val_delta",
        "jac_frob_mean_delta",
    ]
    lines.append("| " + " | ".join(delta_cols) + " |")
    lines.append("|" + "|".join(["---"] + ["---:" for _ in delta_cols[1:]]) + "|")
    for r in trend_rows:
        lines.append("| " + " | ".join(fmt(r.get(c)) for c in delta_cols) + " |")

    lines.append("")
    lines.append("## Reading Guide")
    lines.append("")
    lines.append("- `val_acc_delta > 0`이면 epoch이 늘면서 검증 정확도가 오른 것이다.")
    lines.append("- `val_loss_delta < 0`이면 검증 손실이 줄어든 것이다.")
    lines.append("- `curvature_min_fraction`이 1에 가까워지면 곡률 collapse가 심해진 것이다.")
    lines.append("- `corr_complexity_abs_c_val`이 양수로 커지면 복잡도와 곡률 연결이 강해진 것이다.")
    lines.append("- `jac_frob_mean`이 지나치게 커지면 좌표 변환 반응이 폭주할 수 있다.")
    lines.append("")

    md_text = "\n".join(lines) + "\n"
    md_path.write_text(md_text, encoding="utf-8")
    ubai_md_path.write_text(md_text, encoding="utf-8")

    print(md_path)
    print(json_path)
    print(ubai_md_path)
    print(ubai_json_path)
    print()
    print(md_text)

if __name__ == "__main__":
    main()
