from pathlib import Path
import json
import re
from datetime import datetime

TASKS = ["mrpc", "rte"]
EXPERIMENTS = [
    "main",
    "bert_no_jacobian_reg",
    "bert_no_curvature_aux",
    "bert_no_aux_all",
]

def latest(pattern):
    files = sorted(Path("reports/final").glob(pattern))
    if not files:
        return None
    return files[-1]

def read_json(path):
    if path is None:
        return None
    return json.loads(Path(path).read_text(encoding="utf-8"))

def rows_from_payload(payload):
    if payload is None:
        return []
    return payload.get("rows", [])

def label_majority(task):
    files = sorted(Path("reports/final").glob(f"{task}_label_distribution_*.json"))
    if not files:
        return None
    data = json.loads(files[-1].read_text(encoding="utf-8"))
    for r in data:
        if r.get("split") == "val":
            return r.get("majority_acc")
    return None

def fmt(v):
    if isinstance(v, float):
        return f"{v:.6g}"
    if v is None:
        return ""
    return str(v)

def main():
    summary_rows = []

    for task in TASKS:
        epoch7_path = latest(f"{task}_epoch7_task_ablation_comparison_*.json")
        payload = read_json(epoch7_path)
        rows = rows_from_payload(payload)
        majority = label_majority(task)

        for r in rows:
            val_acc = r.get("val_acc")
            val_acc_minus_majority = None
            if isinstance(val_acc, (int, float)) and isinstance(majority, (int, float)):
                val_acc_minus_majority = val_acc - majority

            summary_rows.append({
                "task": task,
                "experiment": r.get("experiment"),
                "val_acc": val_acc,
                "val_loss": r.get("val_loss"),
                "val_majority_acc": majority,
                "val_acc_minus_majority": val_acc_minus_majority,
                "curvature_min_fraction": r.get("curvature_min_fraction"),
                "curvature_collapse_penalty": r.get("curvature_collapse_penalty"),
                "pred_abs_c_mean": r.get("pred_abs_c_mean"),
                "corr_complexity_abs_c_val": r.get("corr_complexity_abs_c_val"),
                "corr_complexity_jac_val": r.get("corr_complexity_jac_val"),
                "jac_frob_mean": r.get("jac_frob_mean"),
                "source_epoch7_json": str(epoch7_path) if epoch7_path else None,
            })

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("reports/final")
    out_dir.mkdir(parents=True, exist_ok=True)

    md_path = out_dir / f"task_expansion_mrpc_rte_epoch7_summary_{ts}.md"
    json_path = out_dir / f"task_expansion_mrpc_rte_epoch7_summary_{ts}.json"

    json_path.write_text(
        json.dumps({"rows": summary_rows}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    cols = [
        "task",
        "experiment",
        "val_acc",
        "val_loss",
        "val_majority_acc",
        "val_acc_minus_majority",
        "curvature_min_fraction",
        "pred_abs_c_mean",
        "corr_complexity_abs_c_val",
        "corr_complexity_jac_val",
        "jac_frob_mean",
    ]

    lines = []
    lines.append("# Task Expansion Summary: MRPC + RTE")
    lines.append("")
    lines.append("## Epoch 7 Summary Table")
    lines.append("")
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "|".join(["---" if c in ["task", "experiment"] else "---:" for c in cols]) + "|")

    for r in summary_rows:
        lines.append("| " + " | ".join(fmt(r.get(c)) for c in cols) + " |")

    lines.append("")
    lines.append("## Reading")
    lines.append("")
    lines.append("- `val_acc_minus_majority`가 클수록 다수 라벨만 찍는 기준선보다 실제로 더 학습한 것이다.")
    lines.append("- `curvature_min_fraction`이 1에 가까우면 곡률이 최소값으로 몰린 것이다.")
    lines.append("- `jac_frob_mean`이 지나치게 크면 좌표 변환 반응이 폭주한 것으로 볼 수 있다.")
    lines.append("- MRPC는 main이 다수 라벨 기준선을 크게 넘는지, RTE는 작은 validation 크기 때문에 seed 반복이 필요한지를 함께 본다.")
    lines.append("")

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(md_path)
    print(json_path)
    print()
    print(md_path.read_text(encoding="utf-8"))

if __name__ == "__main__":
    main()
