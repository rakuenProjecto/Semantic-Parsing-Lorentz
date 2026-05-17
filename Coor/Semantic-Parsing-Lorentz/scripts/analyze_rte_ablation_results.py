from pathlib import Path
import json
from datetime import datetime

EXPS = ["main", "bert_no_jacobian_reg", "bert_no_curvature_aux", "bert_no_aux_all"]

def latest(pattern):
    files = sorted(Path("reports/final").glob(pattern))
    if not files:
        raise FileNotFoundError(pattern)
    return files[-1]

def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def fmt(x):
    if isinstance(x, float):
        return f"{x:.6g}"
    if x is None:
        return "NA"
    return str(x)

def get_rows(payload):
    if "rows" in payload:
        return payload["rows"]
    if "trend_rows" in payload:
        return payload["trend_rows"]
    raise KeyError("No rows/trend_rows in payload")

def by_exp(rows):
    return {r["experiment"]: r for r in rows}

def main():
    epoch7_path = latest("rte_epoch7_task_ablation_comparison_*.json")
    trend_path = latest("rte_epoch3_to_epoch7_trend_*.json")
    label_path = latest("rte_label_distribution_*.json")

    epoch7 = read_json(epoch7_path)
    trend = read_json(trend_path)
    labels = read_json(label_path)

    rows7 = get_rows(epoch7)
    rows7_by = by_exp(rows7)
    trend_rows = trend["trend_rows"]
    trend_by = by_exp(trend_rows)

    val_majority = None
    for r in labels:
        if r["split"] == "val":
            val_majority = r["majority_acc"]

    main_r = rows7_by["main"]
    no_jac = rows7_by["bert_no_jacobian_reg"]
    no_curv = rows7_by["bert_no_curvature_aux"]
    no_all = rows7_by["bert_no_aux_all"]

    best_acc = max(rows7, key=lambda r: r.get("val_acc") if r.get("val_acc") is not None else -1)
    best_loss = min(rows7, key=lambda r: r.get("val_loss") if r.get("val_loss") is not None else 10**9)

    main_gain_majority = None
    if main_r.get("val_acc") is not None and val_majority is not None:
        main_gain_majority = main_r["val_acc"] - val_majority

    no_all_gain_majority = None
    if no_all.get("val_acc") is not None and val_majority is not None:
        no_all_gain_majority = no_all["val_acc"] - val_majority

    main_vs_no_all = None
    if main_r.get("val_acc") is not None and no_all.get("val_acc") is not None:
        main_vs_no_all = main_r["val_acc"] - no_all["val_acc"]

    jac_ratio = None
    if main_r.get("jac_frob_mean") and no_jac.get("jac_frob_mean"):
        jac_ratio = no_jac["jac_frob_mean"] / main_r["jac_frob_mean"]

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_md = Path("reports/final") / f"rte_ablation_numeric_analysis_{ts}.md"
    out_json = Path("reports/final") / f"rte_ablation_numeric_analysis_{ts}.json"

    payload = {
        "epoch7_path": str(epoch7_path),
        "trend_path": str(trend_path),
        "label_path": str(label_path),
        "val_majority": val_majority,
        "best_acc_experiment": best_acc["experiment"],
        "best_loss_experiment": best_loss["experiment"],
        "main_gain_majority": main_gain_majority,
        "no_all_gain_majority": no_all_gain_majority,
        "main_vs_no_aux_all_val_acc": main_vs_no_all,
        "no_jacobian_vs_main_jac_frob_ratio": jac_ratio,
        "rows_epoch7": rows7,
        "trend_rows": trend_rows,
    }

    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = []
    lines.append("# RTE Ablation Numeric Analysis")
    lines.append("")
    lines.append("## Source Files")
    lines.append("")
    lines.append(f"- epoch7: `{epoch7_path}`")
    lines.append(f"- trend: `{trend_path}`")
    lines.append(f"- label_distribution: `{label_path}`")
    lines.append("")
    lines.append("## Epoch 7 Summary")
    lines.append("")
    lines.append("| experiment | val_acc | val_loss | curvature_min_fraction | pred_abs_c_mean | corr_complexity_abs_c_val | corr_complexity_jac_val | jac_frob_mean |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows7:
        lines.append(
            "| "
            + " | ".join([
                r["experiment"],
                fmt(r.get("val_acc")),
                fmt(r.get("val_loss")),
                fmt(r.get("curvature_min_fraction")),
                fmt(r.get("pred_abs_c_mean")),
                fmt(r.get("corr_complexity_abs_c_val")),
                fmt(r.get("corr_complexity_jac_val")),
                fmt(r.get("jac_frob_mean")),
            ])
            + " |"
        )

    lines.append("")
    lines.append("## Epoch 3 → 7 Change")
    lines.append("")
    lines.append("| experiment | val_acc_delta | val_loss_delta | pred_abs_c_mean_delta | corr_abs_c_delta | corr_jac_delta | jac_frob_delta |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for r in trend_rows:
        lines.append(
            "| "
            + " | ".join([
                r["experiment"],
                fmt(r.get("val_acc_delta")),
                fmt(r.get("val_loss_delta")),
                fmt(r.get("pred_abs_c_mean_delta")),
                fmt(r.get("corr_complexity_abs_c_val_delta")),
                fmt(r.get("corr_complexity_jac_val_delta")),
                fmt(r.get("jac_frob_mean_delta")),
            ])
            + " |"
        )

    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append(f"- RTE validation 다수 라벨 기준선은 `{fmt(val_majority)}`이다.")
    lines.append(f"- epoch 7에서 가장 높은 val_acc는 `{best_acc['experiment']}`의 `{fmt(best_acc.get('val_acc'))}`이다.")
    lines.append(f"- epoch 7에서 가장 낮은 val_loss는 `{best_loss['experiment']}`의 `{fmt(best_loss.get('val_loss'))}`이다.")

    if main_gain_majority is not None:
        lines.append(f"- main은 다수 라벨 기준선보다 `{fmt(main_gain_majority)}`만큼 높다.")
    if no_all_gain_majority is not None:
        lines.append(f"- no_aux_all은 다수 라벨 기준선보다 `{fmt(no_all_gain_majority)}`만큼 높다.")
    if main_vs_no_all is not None:
        lines.append(f"- main과 no_aux_all의 val_acc 차이는 `{fmt(main_vs_no_all)}`이다.")
    if jac_ratio is not None:
        lines.append(f"- no_jacobian_reg의 jac_frob_mean은 main 대비 약 `{fmt(jac_ratio)}`배이다.")

    lines.append("")
    lines.append("### Reading")
    lines.append("")
    lines.append("- main이 다수 라벨 기준선을 뚜렷하게 넘고, no_aux_all보다 높으면 구조 손실이 task 성능에도 기여한 근거가 된다.")
    lines.append("- no_jacobian_reg의 jac_frob_mean이 main보다 크게 높으면 Jacobian 안정화 항이 좌표 변환 폭주를 막는 역할을 한 것으로 해석할 수 있다.")
    lines.append("- no_curvature_aux 또는 no_aux_all의 pred_abs_c_mean이 작고 corr 값이 약하면 곡률 표현이 task 구조와 잘 연결되지 못한 것으로 볼 수 있다.")
    lines.append("- RTE는 validation 샘플 수가 277개로 작기 때문에, 최종 주장에는 seed 반복이 필요하다.")

    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(out_md)
    print(out_json)
    print()
    print(out_md.read_text(encoding="utf-8"))

if __name__ == "__main__":
    main()
