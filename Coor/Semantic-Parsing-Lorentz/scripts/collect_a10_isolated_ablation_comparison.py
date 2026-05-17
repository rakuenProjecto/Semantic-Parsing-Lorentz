from pathlib import Path
import argparse
import json
import re
import shutil
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

def parse_epoch3_val(log_path: Path):
    if not log_path.exists():
        return None, None, None, None

    text = log_path.read_text(encoding="utf-8", errors="replace")
    pattern = re.compile(
        r"epoch=3\s+train_loss=(?P<train_loss>[-+0-9.eE]+)\s+"
        r"train_acc=(?P<train_acc>[-+0-9.eE]+)\s+"
        r"val_loss=(?P<val_loss>[-+0-9.eE]+)\s+"
        r"val_acc=(?P<val_acc>[-+0-9.eE]+)"
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

def root_score(root: Path):
    count = 0
    latest_mtime = root.stat().st_mtime
    for exp in EXPERIMENTS:
        diag = root / exp / "diagnostics_epoch_3.json"
        if diag.exists():
            count += 1
            latest_mtime = max(latest_mtime, diag.stat().st_mtime)
    return count, latest_mtime

def find_latest_root(base: Path):
    candidates = [p for p in base.glob("fresh_a10_isolated_*") if p.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No fresh_a10_isolated_* directory found under {base}")

    # 4개 실험이 모두 끝난 root를 우선 선택. 없으면 가장 많이 완료된 최신 root 선택.
    candidates = sorted(candidates, key=lambda p: root_score(p), reverse=True)
    return candidates[0]

def make_row(exp: str, out_root: Path, log_root: Path, report_root: Path):
    output_dir = out_root / exp
    log_path = log_root / f"{exp}.log"
    diag_path = output_dir / "diagnostics_epoch_3.json"
    summary_path = report_root / exp / "latest_experiment_summary.md"

    diag = read_json(diag_path)
    train_loss, train_acc, val_loss, val_acc = parse_epoch3_val(log_path)

    return {
        "experiment": exp,
        "train_acc": train_acc,
        "val_acc": val_acc,
        "val_loss": val_loss,
        "curvature_min_fraction": diag.get("curvature_min_fraction"),
        "curvature_max_fraction": diag.get("curvature_max_fraction"),
        "curvature_collapse_penalty": diag.get("curvature_collapse_penalty"),
        "curvature_mean_to_min_ratio": diag.get("curvature_mean_to_min_ratio"),
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
        "summary_path": str(summary_path) if summary_path.exists() else None,
    }

def fmt(v):
    if isinstance(v, float):
        return f"{v:.6g}"
    if v is None:
        return ""
    return str(v)

def interpretation(rows):
    by_name = {r["experiment"]: r for r in rows}
    lines = []

    main = by_name.get("main", {})
    no_jac = by_name.get("bert_no_jacobian_reg", {})
    no_curv = by_name.get("bert_no_curvature_aux", {})
    no_all = by_name.get("bert_no_aux_all", {})

    lines.append("## Interpretation")
    lines.append("")

    if main:
        lines.append(
            f"- main은 val_acc={fmt(main.get('val_acc'))}, "
            f"curvature_min_fraction={fmt(main.get('curvature_min_fraction'))}, "
            f"corr_complexity_abs_c_val={fmt(main.get('corr_complexity_abs_c_val'))}를 보인다."
        )

    if no_jac:
        lines.append(
            f"- bert_no_jacobian_reg는 val_acc={fmt(no_jac.get('val_acc'))}, "
            f"jac_frob_mean={fmt(no_jac.get('jac_frob_mean'))}이다. "
            "main보다 jac_frob_mean이 커지면 Jacobian regularization이 좌표 변환 반응을 안정화하는 역할로 해석할 수 있다."
        )

    if no_curv:
        lines.append(
            f"- bert_no_curvature_aux는 val_acc={fmt(no_curv.get('val_acc'))}, "
            f"corr_complexity_abs_c_val={fmt(no_curv.get('corr_complexity_abs_c_val'))}이다. "
            "성능이나 복잡도-곡률 상관이 크게 떨어지면 곡률 보조 손실의 필요성을 보여준다."
        )

    if no_all:
        lines.append(
            f"- bert_no_aux_all은 val_acc={fmt(no_all.get('val_acc'))}, "
            f"curvature_min_fraction={fmt(no_all.get('curvature_min_fraction'))}, "
            f"pred_abs_c_mean={fmt(no_all.get('pred_abs_c_mean'))}이다. "
            "정확도가 유지되더라도 curvature_min_fraction이 1에 가까우면 곡률 표현은 collapse된 것이다."
        )

    lines.append("")
    lines.append(
        "핵심은 accuracy만 보지 않고, curvature_min_fraction, "
        "corr_complexity_abs_c_val, corr_complexity_jac_val, jac_frob_mean을 함께 봐야 한다는 점이다."
    )
    return lines

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        default=None,
        help="예: outputs_a10_isolated/ablations/fresh_a10_isolated_n054.hpc_20260517_164753",
    )
    parser.add_argument("--logs-base", default="logs_a10_isolated/ablations")
    parser.add_argument("--outputs-base", default="outputs_a10_isolated/ablations")
    parser.add_argument("--reports-base", default="reports_a10_isolated/ablations")
    parser.add_argument("--github-reports-dir", default="reports/final")
    parser.add_argument("--ubai-reports-dir", default="reports_a10_isolated/final")
    args = parser.parse_args()

    outputs_base = Path(args.outputs_base)

    if args.output_root:
        out_root = Path(args.output_root)
    else:
        out_root = find_latest_root(outputs_base)

    run_tag = out_root.name
    log_root = Path(args.logs_base) / run_tag
    report_root = Path(args.reports_base) / run_tag

    rows = [make_row(exp, out_root, log_root, report_root) for exp in EXPERIMENTS]

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    github_dir = Path(args.github_reports_dir)
    ubai_dir = Path(args.ubai_reports_dir)
    github_dir.mkdir(parents=True, exist_ok=True)
    ubai_dir.mkdir(parents=True, exist_ok=True)

    stem = f"a10_isolated_sst2_ablation_comparison_{run_tag}_{ts}"
    github_json = github_dir / f"{stem}.json"
    github_md = github_dir / f"{stem}.md"
    ubai_json = ubai_dir / f"{stem}.json"
    ubai_md = ubai_dir / f"{stem}.md"

    payload = {
        "run_tag": run_tag,
        "output_root": str(out_root),
        "log_root": str(log_root),
        "report_root": str(report_root),
        "rows": rows,
    }

    github_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

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
    lines.append("# A10 Isolated SST-2 Ablation Comparison")
    lines.append("")
    lines.append(f"- run_tag: `{run_tag}`")
    lines.append(f"- output_root: `{out_root}`")
    lines.append(f"- log_root: `{log_root}`")
    lines.append(f"- report_root: `{report_root}`")
    lines.append("")
    lines.append("## Summary Table")
    lines.append("")
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "|".join(["---"] + ["---:" for _ in cols[1:]]) + "|")

    for r in rows:
        lines.append("| " + " | ".join(fmt(r.get(c)) for c in cols) + " |")

    lines.append("")
    lines.extend(interpretation(rows))
    lines.append("")
    lines.append("## Paths")
    lines.append("")

    for r in rows:
        lines.append(f"### {r['experiment']}")
        lines.append(f"- output_dir: `{r['output_dir']}`")
        lines.append(f"- log_path: `{r['log_path']}`")
        lines.append(f"- diagnostics_path: `{r['diagnostics_path']}`")
        lines.append(f"- summary_path: `{r['summary_path']}`")
        lines.append("")

    github_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    shutil.copy2(github_json, ubai_json)
    shutil.copy2(github_md, ubai_md)

    print(github_md)
    print(github_json)
    print(ubai_md)
    print(ubai_json)
    print()
    print(github_md.read_text(encoding="utf-8"))

if __name__ == "__main__":
    main()
