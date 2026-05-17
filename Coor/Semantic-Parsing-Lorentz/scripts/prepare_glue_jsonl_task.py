from pathlib import Path
import argparse
import json

from datasets import load_dataset


TASK_FIELDS = {
    "mrpc": ("sentence1", "sentence2", "label"),
    "rte": ("sentence1", "sentence2", "label"),
    "qqp": ("question1", "question2", "label"),
    "qnli": ("question", "sentence", "label"),
    "sst2": ("sentence", None, "label"),
}


def normalize_text(x):
    if x is None:
        return ""
    return str(x).replace("\n", " ").strip()


def write_jsonl(ds, split, out_path, field1, field2, label_field):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n = 0
    with out_path.open("w", encoding="utf-8") as f:
        for ex in ds[split]:
            label = ex.get(label_field)
            if label is None or int(label) < 0:
                continue

            t1 = normalize_text(ex.get(field1))
            if field2 is None:
                text = t1
            else:
                t2 = normalize_text(ex.get(field2))
                text = f"{t1} [SEP] {t2}"

            item = {
                "text": text,
                "label": int(label),
            }
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            n += 1

    return n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="mrpc", choices=sorted(TASK_FIELDS))
    parser.add_argument("--out-root", default="data/real/glue")
    args = parser.parse_args()

    task = args.task.lower()
    field1, field2, label_field = TASK_FIELDS[task]

    print(f"[INFO] loading GLUE task: {task}")
    ds = load_dataset("glue", task)

    out_dir = Path(args.out_root) / task
    train_path = out_dir / "train.jsonl"
    val_path = out_dir / "val.jsonl"

    train_n = write_jsonl(ds, "train", train_path, field1, field2, label_field)

    val_split = "validation"
    if val_split not in ds:
        raise RuntimeError(f"No validation split for task={task}")

    val_n = write_jsonl(ds, val_split, val_path, field1, field2, label_field)

    print(f"[DONE] task={task}")
    print(f"train_jsonl={train_path} num_train={train_n}")
    print(f"val_jsonl={val_path} num_val={val_n}")

    meta = {
        "task": task,
        "train_jsonl": str(train_path),
        "val_jsonl": str(val_path),
        "num_train": train_n,
        "num_val": val_n,
        "field1": field1,
        "field2": field2,
        "label_field": label_field,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
