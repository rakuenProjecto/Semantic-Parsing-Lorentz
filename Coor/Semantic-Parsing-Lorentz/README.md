# Semantic-Parsing-Lorentz

PyTorch scaffold for semantic parsing with instance-wise Lorentz curvature and
a cross-curvature bilinear distance.

## Structure

```text
.
├── configs/default.yaml
├── requirements.txt
├── train.py
├── src/
│   ├── dataset.py
│   ├── distance.py
│   └── model.py
├── checkpoints/
└── logs/
```

## Local Smoke Run

The default config uses the offline dummy encoder, so it does not need a
HuggingFace download:

```bash
pip install -r requirements.txt
python train.py --epochs 1 --num_train_samples 32 --num_val_samples 16 --batch_size 8
```

## Remote HuggingFace Run

```bash
python train.py \
  --no_dummy_encoder \
  --encoder_name bert-base-uncased \
  --batch_size 32 \
  --learning_rate 2e-5 \
  --epochs 10 \
  --output_dir outputs/ubai_bert_lorentz
```

The best checkpoint is saved as `best_model.pth` inside `output_dir`.
