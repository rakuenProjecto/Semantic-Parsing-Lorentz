# Semantic-Parsing-Lorentz

PyTorch research scaffold for semantic parsing with instance-wise Lorentz
curvature, optional linguistic complexity supervision, and an optional true
autograd-Jacobian local metric.

## Structure

```text
.
|-- configs/
|-- requirements.txt
|-- train.py
|-- src/
|   |-- complexity.py
|   |-- dataset.py
|   |-- distance.py
|   |-- model.py
|   `-- true_jacobian.py
|-- tests/
`-- scripts/
```

## Linguistic Complexity Estimator

The mock dataset still supports synthetic complexity values for offline smoke
tests. Set `complexity_mode: heuristic` to compute complexity from text with
regex and lexical rules. Optional `complexity_mode: spacy` uses spaCy POS and
dependency labels when spaCy and an English model are installed; otherwise it
falls back to the heuristic estimator.

The heuristic estimator includes token count, likely verb count, approximate
verb argument count, noun count, adjective count, adverb count, preposition
count, conjunction and subordinate markers, named entity cues, and punctuation.
The verb argument count is approximate and is not a gold syntactic parse.

## True Jacobian Metric

This repository supports a true autograd Jacobian for the Lorentz projection
head:

```text
J_x = d x / d h
```

where `h` is the CLS/text embedding and `x` is the Lorentz point. The Jacobian
is computed only for the mapping `cls_embedding -> tangent/curvature ->
lorentz point`; it does not compute a full input-token or BERT-sequence
Jacobian.

The local metric is:

```text
C_x = J_x J_x^T
```

and the bilinear distance term becomes:

```text
scale = (C_x x)^T W y
```

This is not a learned Jacobian adapter. It is an exact autograd Jacobian for the
projection head and is computationally expensive, especially with
`create_graph: true`. Use smaller batch sizes.

## Commands

Offline smoke test:

```bash
python train.py --epochs 1 --num_train_samples 32 --num_val_samples 16 --batch_size 8
```

True Jacobian smoke test:

```bash
python train.py --config configs/tiny_true_jacobian_smoke.yaml
```

True Jacobian with heuristic complexity:

```bash
python train.py \
  --epochs 1 \
  --num_train_samples 32 \
  --num_val_samples 16 \
  --batch_size 4 \
  --complexity_mode heuristic \
  --use_true_jacobian_metric \
  --true_jacobian_target lorentz
```

UBAI BERT run:

```bash
torchrun --nproc_per_node=4 train.py \
  --config configs/bert_true_jacobian_complexity.yaml \
  --no_dummy_encoder \
  --encoder_name bert-base-uncased \
  --output_dir outputs/ubai_bert_true_jacobian_complexity
```

## JSONL Dataset Format

Use `dataset_type: jsonl`, `train_jsonl`, and optionally `val_jsonl`.

```json
{"text": "Book a direct flight from Seoul to Boston after Monday.", "label": "flight.search"}
{"text": "Will it rain tomorrow in Seoul?", "label": "weather.lookup"}
```

Integer labels are also supported:

```json
{"text": "Book a flight.", "label": 0}
```

If `complexity` is omitted, set `complexity_mode: heuristic` or
`complexity_mode: spacy` to compute it from text.

## Suggested Ablation Study

| Variant | Curvature | Complexity loss | True Jacobian metric | Detached `C_x` | Notes |
| --- | --- | --- | --- | --- | --- |
| BERT classifier baseline | No | No | No | No | Standard encoder classifier |
| Lorentz curvature only | Yes | No | No | No | Existing dynamic-curvature model |
| Lorentz curvature + complexity loss | Yes | Yes | No | No | Tests complexity-curvature coupling |
| Lorentz curvature + true Jacobian metric | Yes | No | Yes | No | Tests local metric correction |
| Lorentz curvature + complexity + true Jacobian metric | Yes | Yes | Yes | No | Full proposed prototype |
| True Jacobian metric with detached `C_x` | Yes | Optional | Yes | Yes | Ablates higher-order gradient path |
| Identity-mix sweep | Yes | Optional | Yes | No | Compare `identity_mix` values |

The best checkpoint is saved as `best_model.pth` inside `output_dir`.
