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

True Jacobian with curvature anti-collapse controls:

```bash
python train.py --config configs/bert_true_jacobian_anti_collapse_debug.yaml
```

UBAI BERT run:

```bash
torchrun --nproc_per_node=4 train.py \
  --config configs/bert_true_jacobian_complexity.yaml \
  --no_dummy_encoder \
  --encoder_name bert-base-uncased \
  --output_dir outputs/ubai_bert_true_jacobian_complexity
```

## UBAI Validation Notes

The following checks have passed on UBAI:

- `python -m pytest tests`
- `python train.py --config configs/tiny_true_jacobian_smoke.yaml`
- `python train.py --epochs 1 --num_train_samples 32 --num_val_samples 16 --batch_size 8`
- BERT true-Jacobian 1 GPU smoke runs with batch sizes 1 and 2
- A longer 1 GPU BERT true-Jacobian run with batch size 2, 3 epochs, and 1024 train samples

Notes from those runs:

- YAML scientific notation such as `2e-5` can be parsed as a string in some environments. `train.py` now normalizes numeric and boolean config values after loading, and BERT configs use plain decimal learning rates.
- Batch-level correlations are not meaningful for tiny batches. Logs now mark batch correlations with validity flags and write full-validation diagnostics to `diagnostics_epoch_{epoch}.json` plus `diagnostics_final.json`.
- The mock dataset is intentionally easy. A validation accuracy of `1.0` on mock data is a functionality check, not paper-level semantic parsing performance.
- Curvature can collapse near `min_abs_curvature`. Treat `mean_abs_c` near `0.05` together with `curvature_min_fraction=1.0` as curvature collapse. Use `pred_abs_c_*`, `target_abs_c_*`, `curvature_min_fraction`, `curvature_mean_to_min_ratio`, `curvature_collapse_penalty`, and `curvature_spread_penalty` in the diagnostics JSON files to monitor this.
- `configs/bert_true_jacobian_anti_collapse.yaml` keeps the true autograd Jacobian path but uses `curvature_parameterization: softplus_floor`, `curvature_init_abs_c`, target scaling, sample-wise floor penalties, raw-floor penalties, and spread penalties. Tune `curvature_anti_collapse_weight`, `curvature_target_scale`, `curvature_min_margin`, and `curvature_spread_weight` only after checking that classification loss and Jacobian diagnostics remain finite.
- True Jacobian remains expensive. Start with `configs/bert_mock_fast_debug.yaml` or `configs/tiny_true_jacobian_smoke.yaml` before full BERT runs.

One-GPU UBAI scripts:

```bash
bash scripts/run_ubai_1gpu_true_jacobian.sh
bash scripts/run_ubai_1gpu_true_jacobian_anti_collapse.sh
bash scripts/run_ubai_1gpu_no_jacobian.sh
bash scripts/run_ubai_1gpu_detached_jacobian.sh
```

Parse a log to CSV:

```bash
python scripts/parse_training_log.py --log logs/bert_true_jacobian_1gpu_bs2_e3_n1024.log
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
