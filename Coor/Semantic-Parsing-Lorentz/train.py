"""Train the dynamic-curvature Lorentz semantic parser."""

from __future__ import annotations

import argparse
import logging
import random
from functools import partial
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from src.dataset import MockSemanticParsingDataset, SimpleWhitespaceTokenizer, semantic_collate_fn
from src.model import SemanticLorentzParser


TensorLike = Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Lorentz semantic parser")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config")

    parser.add_argument("--seed", type=int)
    parser.add_argument("--use_dummy_encoder", dest="use_dummy_encoder", action="store_true", default=None)
    parser.add_argument("--no_dummy_encoder", dest="use_dummy_encoder", action="store_false")
    parser.add_argument("--encoder_name", type=str)
    parser.add_argument("--hf_local_files_only", dest="hf_local_files_only", action="store_true", default=None)
    parser.add_argument("--no_hf_local_files_only", dest="hf_local_files_only", action="store_false")
    parser.add_argument("--freeze_encoder", dest="freeze_encoder", action="store_true", default=None)
    parser.add_argument("--no_freeze_encoder", dest="freeze_encoder", action="store_false")

    parser.add_argument("--num_labels", type=int)
    parser.add_argument("--hidden_dim", type=int)
    parser.add_argument("--tangent_dim", type=int)
    parser.add_argument("--dummy_vocab_size", type=int)
    parser.add_argument("--max_position_embeddings", type=int)
    parser.add_argument("--max_length", type=int)
    parser.add_argument("--dropout", type=float)

    parser.add_argument("--min_abs_curvature", type=float)
    parser.add_argument("--max_abs_curvature", type=float)
    parser.add_argument("--max_expmap_norm", type=float)
    parser.add_argument("--max_tangent_norm", type=float)
    parser.add_argument("--metric_delta_scale", type=float)
    parser.add_argument("--metric_reg_weight", type=float)
    parser.add_argument("--curvature_aux_weight", type=float)

    parser.add_argument("--num_train_samples", type=int)
    parser.add_argument("--num_val_samples", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--num_workers", type=int)

    parser.add_argument("--epochs", type=int)
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--weight_decay", type=float)
    parser.add_argument("--grad_clip_norm", type=float)
    parser.add_argument("--log_interval", type=int)
    parser.add_argument("--output_dir", type=str)
    return parser.parse_args()


def load_config(config_path: str, cli_args: argparse.Namespace) -> Dict[str, Any]:
    path = Path(config_path)
    config: Dict[str, Any] = {}
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}
            config.update(loaded)

    for key, value in vars(cli_args).items():
        if key != "config" and value is not None:
            config[key] = value
    return config


def setup_logging(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("lorentz_semantic_parser")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(output_dir / "training.log", encoding="utf-8")
    file_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    return logger


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_tokenizer(config: Dict[str, Any]) -> Any:
    if config["use_dummy_encoder"]:
        return SimpleWhitespaceTokenizer(vocab_size=config["dummy_vocab_size"])

    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise ImportError("Install transformers or set use_dummy_encoder=true") from exc

    return AutoTokenizer.from_pretrained(
        config["encoder_name"],
        local_files_only=config.get("hf_local_files_only", False),
    )


def build_dataloader(
    dataset: MockSemanticParsingDataset,
    tokenizer: Any,
    config: Dict[str, Any],
    shuffle: bool,
    device: torch.device,
) -> DataLoader:
    collate = partial(semantic_collate_fn, tokenizer=tokenizer, max_length=config["max_length"])
    return DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=shuffle,
        num_workers=config["num_workers"],
        pin_memory=device.type == "cuda",
        collate_fn=collate,
    )


def move_batch_to_device(batch: Dict[str, TensorLike], device: torch.device) -> Dict[str, TensorLike]:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def run_epoch(
    model: SemanticLorentzParser,
    dataloader: DataLoader,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    config: Dict[str, Any],
    logger: logging.Logger,
    epoch: int,
) -> Dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for step, batch in enumerate(dataloader, start=1):
        batch = move_batch_to_device(batch, device)
        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            output = model(
                input_ids=batch["input_ids"],
                attention_mask=batch.get("attention_mask"),
                labels=batch["labels"],
            )
            loss = output["loss"]
            if is_train and config["metric_reg_weight"] > 0:
                loss = loss + config["metric_reg_weight"] * model.metric_regularization()
            if (
                is_train
                and config.get("curvature_aux_weight", 0.0) > 0
                and "complexity" in batch
            ):
                complexity = batch["complexity"].view(-1, 1)
                normalized_complexity = ((complexity - 1.0) / 3.0).clamp(0.0, 1.0)
                curvature_span = config["max_abs_curvature"] - config["min_abs_curvature"]
                target_abs_curvature = config["min_abs_curvature"] + curvature_span * normalized_complexity
                curvature_loss = F.mse_loss(-output["curvature"], target_abs_curvature)
                loss = loss + config["curvature_aux_weight"] * curvature_loss

            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at epoch={epoch}, step={step}")

            if is_train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip_norm"])
                optimizer.step()

        batch_size = batch["labels"].size(0)
        total_loss += loss.item() * batch_size
        predictions = output["logits"].argmax(dim=-1)
        total_correct += (predictions == batch["labels"]).sum().item()
        total_count += batch_size

        if is_train and step % config["log_interval"] == 0:
            avg_loss = total_loss / max(total_count, 1)
            avg_acc = total_correct / max(total_count, 1)
            curvature = output["curvature"].detach().mean().item()
            logger.info(
                "epoch=%d step=%d loss=%.4f acc=%.4f mean_c=%.4f",
                epoch,
                step,
                avg_loss,
                avg_acc,
                curvature,
            )

    return {
        "loss": total_loss / max(total_count, 1),
        "accuracy": total_correct / max(total_count, 1),
    }


def save_checkpoint(
    path: Path,
    model: SemanticLorentzParser,
    optimizer: torch.optim.Optimizer,
    config: Dict[str, Any],
    epoch: int,
    metrics: Dict[str, float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
            "metrics": metrics,
        },
        path,
    )


def main() -> None:
    args = parse_args()
    config = load_config(args.config, args)
    output_dir = Path(config["output_dir"])
    logger = setup_logging(output_dir)

    set_seed(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("using device=%s", device)

    with (output_dir / "config_used.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=True)

    tokenizer = build_tokenizer(config)
    train_dataset = MockSemanticParsingDataset(
        num_samples=config["num_train_samples"],
        num_labels=config["num_labels"],
        seed=config["seed"],
    )
    val_dataset = MockSemanticParsingDataset(
        num_samples=config["num_val_samples"],
        num_labels=config["num_labels"],
        seed=config["seed"] + 1,
    )
    train_loader = build_dataloader(train_dataset, tokenizer, config, shuffle=True, device=device)
    val_loader = build_dataloader(val_dataset, tokenizer, config, shuffle=False, device=device)

    model = SemanticLorentzParser(
        num_labels=config["num_labels"],
        encoder_name=config["encoder_name"],
        use_dummy_encoder=config["use_dummy_encoder"],
        dummy_vocab_size=config["dummy_vocab_size"],
        hidden_dim=config["hidden_dim"],
        tangent_dim=config["tangent_dim"],
        max_position_embeddings=config["max_position_embeddings"],
        dropout=config["dropout"],
        freeze_encoder=config["freeze_encoder"],
        min_abs_curvature=config["min_abs_curvature"],
        max_abs_curvature=config["max_abs_curvature"],
        max_expmap_norm=config["max_expmap_norm"],
        max_tangent_norm=config["max_tangent_norm"],
        metric_delta_scale=config["metric_delta_scale"],
        hf_local_files_only=config.get("hf_local_files_only", False),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )

    best_val_loss = float("inf")
    best_path = output_dir / "best_model.pth"
    logger.info("labels=%s", train_dataset.label_names)

    for epoch in range(1, config["epochs"] + 1):
        train_metrics = run_epoch(model, train_loader, optimizer, device, config, logger, epoch)
        val_metrics = run_epoch(model, val_loader, None, device, config, logger, epoch)
        logger.info(
            "epoch=%d train_loss=%.4f train_acc=%.4f val_loss=%.4f val_acc=%.4f",
            epoch,
            train_metrics["loss"],
            train_metrics["accuracy"],
            val_metrics["loss"],
            val_metrics["accuracy"],
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            save_checkpoint(best_path, model, optimizer, config, epoch, val_metrics)
            logger.info("saved new best checkpoint to %s", best_path)

    logger.info("training complete best_val_loss=%.4f checkpoint=%s", best_val_loss, best_path)


if __name__ == "__main__":
    main()
