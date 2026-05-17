from pathlib import Path
import json
import re
from datetime import datetime

MAIN = {
    "experiment": "main",
    "output_dir": Path("outputs/real_sst2_full_e10_20260517_014513"),
    "log_path": Path("logs/real_sst2_full_e10_20260517_014513.log"),
    "diag_path": Path("outputs/real_sst2_full_e10_20260517_014513/diagnostics_epoch_3.json"),
}

ABLATIONS = [
    {
        "experiment": "bert_no_jacobian_reg",
        "output_dir": Path("outputs/ablations/bert_no_jacobian_reg_20260517_103225"),
        "log_path": Path("logs/ablations/bert_no_jacobian_reg_20260517_103225.log"),
        "diag_path": Path("outputs/ablations/bert_no_jacobian_reg_20260517_103225/diagnostics_epoch_3.json"),
    },
    {
        "experiment": "bert_no_curvature_aux",
        "output_dir": Path("outputs/ablations/bert_no_curvature_aux_20260517_103225"),
        "log_path": Path("logs/ablations/bert_no_curvature_aux_20260517_103225.log"),
        "diag_path": Path("outputs/ablations/bert_no_curvature_aux_20260517_103225/diagnostics_epoch_3.json"),
    },
    {
        "experiment": "bert_no_aux_all",
        "output_dir": Path("outputs/ablations/bert_no_aux_all_20260517_103225"),
        "log_path": Path("logs/ablations/bert_no_aux_all_20260517_103225.log"),
        "diag_path": Path("outputs/ablations/bert_no_aux_all_20260517_103225/diagnostics_epoch_3.json"),
    },
]

def load_json(path: Path):
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

def row(item):
    d = load_json(item["diag_path"])
    train_loss, train_acc, val_loss, val_acc = parse_epoch3_val(item["log_path"])

    return {
        "experiment": item["experiment"],
        "train_acc": train_acc,
        "val_acc": val_acc,
        "val_loss": val_loss,
        "curvature_min_fraction": d.get("curvature_min_fraction"),
        "curvature_max_fraction": d.get("curvature_max_fraction"),
        "curvature_collapse_penalty": d.get("curvature_collapse_penalty"),
        "pred_abs_c_mean": d.get("pred_abs_c_mean", d.get("mean_abs_c")),
        "pred_abs_c_std": d.get("pred_abs_c_std", d.get("std_abs_c")),
        "corr_complexity_abs_c_val": d.get("corr_complexity_abs_c"),
        "corr_complexity_jac_val": d.get("corr_complexity_jac"),
        "jac_frob_mean": d.get("jac_frob_mean"),
        "jac_effective_rank_mean": d.get("jac_effective_rank_mean"),
        "output_dir": str(item["output_dir"]),
        "log_path": str(item["log_path"]),
        "diagnostics_path": str(item["diag_path"]),
    }

rows = [row(MAIN)] + [row(x) for x in ABLATIONS]

out_dir = Path("reports/final")
out_dir.mkdir(parents=True, exist_ok=True)

ts = datetime.now().strftime("%Y%m%d_%H%M%S")
json_path = out_dir / f"n001_sst2_ablation_comparison_{ts}.json"
md_path = out_dir / f"n001_sst2_ablation_comparison_{ts}.md"

json_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

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
lines.append("# n001 SST-2 Ablation Comparison")
lines.append("")
lines.append("## Summary Table")
lines.append("")
lines.append("| " + " | ".join(cols) + " |")
lines.append("|" + "|".join(["---"] + ["---:" for _ in cols[1:]]) + "|")

for r in rows:
    values = []
    for c in cols:
        v = r.get(c)
        if isinstance(v, float):
            values.append(f"{v:.6g}")
        elif v is None:
            values.append("")
        else:
            values.append(str(v))
    lines.append("| " + " | ".join(values) + " |")

lines.append("")
lines.append("## Interpretation")
lines.append("")
lines.append("- main은 val_acc=0.9048, curvature_min_fraction=0.0으로 정확도와 곡률 안정성을 동시에 유지했다.")
lines.append("- bert_no_jacobian_reg는 val_acc가 가장 높지만 jac_frob_mean이 크게 증가하여 Jacobian 안정화 항이 좌표 변환 반응을 억제하는 역할을 했을 가능성이 있다.")
lines.append("- bert_no_curvature_aux는 val_acc가 0.5092로 크게 하락하여 곡률 보조 손실 제거가 성능 안정성에 큰 악영향을 준 것으로 보인다.")
lines.append("- bert_no_aux_all은 val_acc는 main과 같지만 curvature_min_fraction=1.0으로 validation 전체가 최소 곡률 근처에 몰렸다. 즉, 정확도만 보면 괜찮아 보여도 곡률 표현은 완전히 collapse되었다.")
lines.append("")
lines.append("## Paths")
lines.append("")

for r in rows:
    lines.append(f"### {r['experiment']}")
    lines.append(f"- output_dir: `{r['output_dir']}`")
    lines.append(f"- log_path: `{r['log_path']}`")
    lines.append(f"- diagnostics_path: `{r['diagnostics_path']}`")
    lines.append("")

md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

print(md_path)
print(json_path)
print()
print(md_path.read_text(encoding="utf-8"))
