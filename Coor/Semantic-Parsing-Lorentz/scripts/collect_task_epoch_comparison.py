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

def find_latest_run_root(outputs_base: Path, task: str):
    candidates = [
        p for p in outputs_base.glob(f"glue_{task}_*")
        if p.is_dir()
    ]
    if not candidates:
        raise FileNotFoundError(f"No run root found under {outputs_base} for task={task}")

    def score(root: Path):
        count = 0
        mtime = root.stat().st_mtime
        for exp in EXPERIMENTS:
            for p in (root / exp).glob("diagnostics_epoch_*.json"):
                count += 1
                mtime = max(mtime, p.stat().st_mtime)
        return count, mtime

    return max(candidates, key=score)

def fmt(v):
    if isinstance(v, float):
        return f"{v:.6g}"
    if v is None:
        return ""
    return str(v)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="mrpc")
    parser.add_argument("--epoch", type=int, default=3)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--outputs-base", default="outputs_task_isolated")
    parser.add_argument("--logs-base", default="logs_task_isolated")
    parser.add_argument("--reports-dir", default="reports/final")
    parser.add_argument("--ubai-reports-dir", default="reports_task_isolated/final")
    args = parser.parse_args()

    task = args.task.lower()
    epoch = args.epoch

    if args.output_root:
        out_root = Path(args.output_root)
    else:
        out_root = find_latest_run_root(Path(args.outputs_base), task)

    run_tag = out_root.name
    log_root = Path(args.logs_base) / run_tag

    rows = []
    for exp in EXPERIMENTS:
        output_dir = out_root / exp
        log_path = log_root / f"{exp}.log"
        diag_path = output_dir / f"diagnostics_epoch_{epoch}.json"

        diag = read_json(diag_path)
        train_loss, train_acc, val_loss, val_acc = parse_epoch_val(log_path, epoch)

        rows.append({
            "experiment": exp,
            "epoch": epoch,
            "train_acc": train_acc,
            "val_acc": val_acc,
            "val_loss": val_loss,
            "curvature_min_fraction": diag.get("curvature_min_fraction"),
            "curvature_max_fraction": diag.get("curvature_max_fraction"),
            "curvature_collapse_penalty": diag.get("curvature_collapse_penalty"),
            "pred_abs_c_mean": diag.get("pred_abs_c_mean", diag.get("mean_abs_c")),
            "pred_abs_c_std": diag.get("pred_abs_c_std", diag.get("std_abs_c")),
            "corr_complexity_abs_c_val": diag.get("corr_complexity_abs_c"),
            "corr_complexity_jac_val": diag.get("corr_complexity_jac"),
            "jac_frob_mean": diag.get("jac_frob_mean"),
            "jac_frob_std": diag.get("jac_frob_std"),
            "jac_effective_rank_mean": diag.get("jac_effective_rank_mean"),
            "output_dir": str(output_dir),
            "log_path": str(log_path),
            "diagnostics_path": str(diag_path),
            "diagnostics_exists": diag_path.exists(),
        })

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    reports_dir = Path(args.reports_dir)
    ubai_reports_dir = Path(args.ubai_reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    ubai_reports_dir.mkdir(parents=True, exist_ok=True)

    stem = f"{task}_epoch{epoch}_task_ablation_comparison_{run_tag}_{ts}"

    json_path = reports_dir / f"{stem}.json"
    md_path = reports_dir / f"{stem}.md"
    ubai_json_path = ubai_reports_dir / f"{stem}.json"
    ubai_md_path = ubai_reports_dir / f"{stem}.md"

    payload = {
        "task": task,
        "epoch": epoch,
        "run_tag": run_tag,
        "output_root": str(out_root),
        "log_root": str(log_root),
        "rows": rows,
    }

    json_text = json.dumps(payload, indent=2, ensure_ascii=False)
    json_path.write_text(json_text, encoding="utf-8")
    ubai_json_path.write_text(json_text, encoding="utf-8")

    cols = [
        "experiment",
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

    lines = []
    lines.append(f"# {task.upper()} Epoch {epoch} Task Ablation Comparison")
    lines.append("")
    lines.append(f"- task: `{task}`")
    lines.append(f"- epoch: `{epoch}`")
    lines.append(f"- run_tag: `{run_tag}`")
    lines.append(f"- output_root: `{out_root}`")
    lines.append(f"- log_root: `{log_root}`")
    lines.append("")
    lines.append("## Summary Table")
    lines.append("")
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "|".join(["---"] + ["---:" for _ in cols[1:]]) + "|")

    for r in rows:
        lines.append("| " + " | ".join(fmt(r.get(c)) for c in cols) + " |")

    lines.append("")
    lines.append("## Quick Interpretation")
    lines.append("")
    lines.append("- `val_acc`는 단순 분류 정확도다.")
    lines.append("- `curvature_min_fraction`이 1에 가까우면 곡률이 최소값으로 무너진 것이다.")
    lines.append("- `corr_complexity_abs_c_val`이 높으면 문장 복잡도와 곡률 크기가 잘 연결된 것이다.")
    lines.append("- `corr_complexity_jac_val`은 문장 복잡도와 Jacobian 반응의 연결 정도다.")
    lines.append("- `jac_frob_mean`이 너무 커지면 좌표 변환 반응이 과하게 커졌을 가능성이 있다.")
    lines.append("")
    lines.append("## Paths")
    lines.append("")

    for r in rows:
        lines.append(f"### {r['experiment']}")
        lines.append(f"- output_dir: `{r['output_dir']}`")
        lines.append(f"- log_path: `{r['log_path']}`")
        lines.append(f"- diagnostics_path: `{r['diagnostics_path']}`")
        lines.append(f"- diagnostics_exists: `{r['diagnostics_exists']}`")
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
