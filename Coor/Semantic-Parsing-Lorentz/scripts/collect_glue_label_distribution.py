from pathlib import Path
import argparse
import json
from collections import Counter
from datetime import datetime

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--reports-dir", default="reports/final")
    args = parser.parse_args()

    task = args.task.lower()
    rows = []

    for split in ["train", "val"]:
        path = Path(f"data/real/glue/{task}/{split}.jsonl")
        labels = []
        with path.open(encoding="utf-8") as f:
            for line in f:
                labels.append(json.loads(line)["label"])

        c = Counter(labels)
        total = sum(c.values())
        majority_acc = max(c.values()) / total if total else None

        rows.append({
            "task": task,
            "split": split,
            "counts": dict(c),
            "total": total,
            "majority_acc": majority_acc,
        })

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.reports_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / f"{task}_label_distribution_{ts}.json"
    md_path = out_dir / f"{task}_label_distribution_{ts}.md"

    json_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = []
    lines.append(f"# {task.upper()} Label Distribution")
    lines.append("")
    lines.append("| split | total | counts | majority_acc |")
    lines.append("|---|---:|---|---:|")
    for r in rows:
        lines.append(
            f"| {r['split']} | {r['total']} | `{r['counts']}` | {r['majority_acc']:.6g} |"
        )

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(md_path)
    print(json_path)
    print(md_path.read_text(encoding="utf-8"))

if __name__ == "__main__":
    main()
