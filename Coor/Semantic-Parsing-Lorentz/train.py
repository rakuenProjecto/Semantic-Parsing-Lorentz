"""Train the dynamic-curvature Lorentz semantic parser."""

from __future__ import annotations

import argparse
import logging
import os
import random
from functools import partial
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset

from src.dataset import (
    JsonlSemanticParsingDataset,
    MockSemanticParsingDataset,
    SimpleWhitespaceTokenizer,
    semantic_collate_fn,
)
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

    parser.add_argument("--dataset_type", choices=["mock", "jsonl"])
    parser.add_argument("--train_jsonl", type=str)
    parser.add_argument("--val_jsonl", type=str)
    parser.add_argument("--complexity_mode", choices=["synthetic", "heuristic", "spacy"])
    parser.add_argument("--return_complexity_features", dest="return_complexity_features", action="store_true", default=None)
    parser.add_argument("--no_return_complexity_features", dest="return_complexity_features", action="store_false")

    parser.add_argument("--use_true_jacobian_metric", dest="use_true_jacobian_metric", action="store_true", default=None)
    parser.add_argument("--no_true_jacobian_metric", dest="use_true_jacobian_metric", action="store_false")
    parser.add_argument("--true_jacobian_target", choices=["lorentz", "tangent"])
    parser.add_argument("--true_jacobian_create_graph", dest="true_jacobian_create_graph", action="store_true", default=None)
    parser.add_argument("--no_true_jacobian_create_graph", dest="true_jacobian_create_graph", action="store_false")
    parser.add_argument("--true_jacobian_normalize", dest="true_jacobian_normalize", action="store_true", default=None)
    parser.add_argument("--no_true_jacobian_normalize", dest="true_jacobian_normalize", action="store_false")
    parser.add_argument("--true_jacobian_identity_mix", type=float)
    parser.add_argument("--true_jacobian_max_batch", type=int)
    parser.add_argument("--true_jacobian_reg_weight", type=float)
    parser.add_argument("--true_jacobian_complexity_weight", type=float)
    parser.add_argument("--detach_true_jacobian_metric", dest="detach_true_jacobian_metric", action="store_true", default=None)
    parser.add_argument("--no_detach_true_jacobian_metric", dest="detach_true_jacobian_metric", action="store_false")
    parser.add_argument("--log_true_jacobian_stats", dest="log_true_jacobian_stats", action="store_true", default=None)
    parser.add_argument("--no_log_true_jacobian_stats", dest="log_true_jacobian_stats", action="store_false")
    parser.add_argument("--true_jacobian_log_interval", type=int)
    parser.add_argument("--save_jacobian_diagnostics", dest="save_jacobian_diagnostics", action="store_true", default=None)
    parser.add_argument("--no_save_jacobian_diagnostics", dest="save_jacobian_diagnostics", action="store_false")

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
    apply_config_defaults(config)
    return config


def apply_config_defaults(config: Dict[str, Any]) -> None:
    defaults = {
        "dataset_type": "mock",
        "train_jsonl": None,
        "val_jsonl": None,
        "complexity_mode": "synthetic",
        "return_complexity_features": False,
        "use_true_jacobian_metric": False,
        "true_jacobian_target": "lorentz",
        "true_jacobian_create_graph": True,
        "true_jacobian_normalize": True,
        "true_jacobian_identity_mix": 0.1,
        "true_jacobian_max_batch": None,
        "true_jacobian_reg_weight": 0.0,
        "true_jacobian_complexity_weight": 0.0,
        "detach_true_jacobian_metric": False,
        "log_true_jacobian_stats": False,
        "true_jacobian_log_interval": 100,
        "save_jacobian_diagnostics": False,
    }
    for key, value in defaults.items():
        config.setdefault(key, value)


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


def get_device() -> torch.device:
    local_rank = os.environ.get("LOCAL_RANK")
    if torch.cuda.is_available():
        if local_rank is not None:
            device = torch.device(f"cuda:{int(local_rank)}")
            torch.cuda.set_device(device)
            return device
        return torch.device("cuda")
    return torch.device("cpu")


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


def build_datasets(config: Dict[str, Any]) -> Tuple[Dataset, Dataset]:
    complexity_mode = config.get("complexity_mode", "synthetic")
    return_features = config.get("return_complexity_features", False)
    if config.get("dataset_type", "mock") == "mock":
        train_dataset = MockSemanticParsingDataset(
            num_samples=config["num_train_samples"],
            num_labels=config["num_labels"],
            seed=config["seed"],
            complexity_mode=complexity_mode,
            return_complexity_features=return_features,
        )
        val_dataset = MockSemanticParsingDataset(
            num_samples=config["num_val_samples"],
            num_labels=config["num_labels"],
            seed=config["seed"] + 1,
            complexity_mode=complexity_mode,
            return_complexity_features=return_features,
        )
        return train_dataset, val_dataset

    if config.get("dataset_type") != "jsonl":
        raise ValueError("dataset_type must be 'mock' or 'jsonl'")
    if not config.get("train_jsonl"):
        raise ValueError("dataset_type=jsonl requires train_jsonl")

    train_dataset = JsonlSemanticParsingDataset(
        config["train_jsonl"],
        complexity_mode=complexity_mode,
        return_complexity_features=return_features,
    )
    val_path = config.get("val_jsonl") or config["train_jsonl"]
    val_dataset = JsonlSemanticParsingDataset(
        val_path,
        complexity_mode=complexity_mode,
        return_complexity_features=return_features,
        label_to_id=train_dataset.label_to_id,
    )
    config["num_labels"] = len(train_dataset.label_names)
    return train_dataset, val_dataset


def build_dataloader(
    dataset: Dataset,
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


def safe_correlation(x: TensorLike, y: TensorLike, eps: float = 1e-8) -> float:
    x_tensor = torch.as_tensor(x).detach().float().view(-1)
    y_tensor = torch.as_tensor(y).detach().float().view(-1)
    if x_tensor.numel() < 2 or y_tensor.numel() < 2:
        return float("nan")
    x_centered = x_tensor - x_tensor.mean()
    y_centered = y_tensor - y_tensor.mean()
    denom = x_centered.norm() * y_centered.norm()
    if denom.item() <= eps:
        return float("nan")
    return float((x_centered * y_centered).sum().div(denom).item())


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
    total_ce_loss = 0.0
    total_correct = 0
    total_count = 0
    diagnostics_saved = False
    true_jacobian_enabled = bool(config.get("use_true_jacobian_metric", False))
    jacobian_log_interval = int(config.get("true_jacobian_log_interval", 100))

    for step, batch in enumerate(dataloader, start=1):
        batch = move_batch_to_device(batch, device)
        if is_train:
            optimizer.zero_grad(set_to_none=True)

        need_diagnostics = (
            true_jacobian_enabled
            and (
                (is_train and bool(config.get("save_jacobian_diagnostics", False)) and not diagnostics_saved)
                or (
                    is_train
                    and bool(config.get("log_true_jacobian_stats", False))
                    and step % max(jacobian_log_interval, 1) == 0
                )
            )
        )

        with torch.set_grad_enabled(is_train or true_jacobian_enabled):
            output = model(
                input_ids=batch["input_ids"],
                attention_mask=batch.get("attention_mask"),
                labels=batch["labels"],
                return_diagnostics=need_diagnostics,
            )
            ce_loss = output["loss"]
            loss = ce_loss
            metric_reg_loss = model.metric_regularization()
            curvature_loss = torch.zeros((), dtype=loss.dtype, device=loss.device)
            jacobian_reg_loss = torch.zeros((), dtype=loss.dtype, device=loss.device)
            jacobian_complexity_loss = torch.zeros((), dtype=loss.dtype, device=loss.device)

            if is_train and config["metric_reg_weight"] > 0:
                loss = loss + config["metric_reg_weight"] * metric_reg_loss
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

            if true_jacobian_enabled and is_train and "jacobian_frobenius" in output:
                jacobian_frobenius = output["jacobian_frobenius"]
                if config.get("true_jacobian_reg_weight", 0.0) > 0:
                    jacobian_reg_loss = torch.log1p(jacobian_frobenius).pow(2).mean()
                    loss = loss + config["true_jacobian_reg_weight"] * jacobian_reg_loss
                if (
                    config.get("true_jacobian_complexity_weight", 0.0) > 0
                    and "complexity" in batch
                ):
                    complexity_norm = ((batch["complexity"] - 1.0) / 3.0).clamp(0.0, 1.0)
                    jac_norm = jacobian_frobenius / (jacobian_frobenius.detach().mean().clamp_min(1e-8))
                    jac_norm_scaled = jac_norm.clamp(0.0, 3.0) / 3.0
                    jacobian_complexity_loss = F.smooth_l1_loss(jac_norm_scaled, complexity_norm)
                    loss = loss + config["true_jacobian_complexity_weight"] * jacobian_complexity_loss

            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at epoch={epoch}, step={step}")

            if is_train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip_norm"])
                optimizer.step()

            if (
                true_jacobian_enabled
                and is_train
                and bool(config.get("save_jacobian_diagnostics", False))
                and not diagnostics_saved
                and "true_jacobian" in output
            ):
                save_jacobian_diagnostics(output, batch, Path(config["output_dir"]) / "jacobian_diagnostics.pt")
                diagnostics_saved = True

        batch_size = batch["labels"].size(0)
        total_loss += loss.item() * batch_size
        total_ce_loss += ce_loss.item() * batch_size
        predictions = output["logits"].argmax(dim=-1)
        total_correct += (predictions == batch["labels"]).sum().item()
        total_count += batch_size

        should_log_step = is_train and (
            step % config["log_interval"] == 0
            or (
                true_jacobian_enabled
                and bool(config.get("log_true_jacobian_stats", False))
                and step % max(jacobian_log_interval, 1) == 0
            )
        )
        if should_log_step:
            avg_loss = total_loss / max(total_count, 1)
            avg_ce_loss = total_ce_loss / max(total_count, 1)
            avg_acc = total_correct / max(total_count, 1)
            mean_abs_curvature = output["curvature"].detach().abs().mean().item()
            mean_complexity = batch["complexity"].detach().mean().item() if "complexity" in batch else float("nan")
            log_message = (
                "epoch=%d step=%d loss=%.4f ce=%.4f acc=%.4f curvature_loss=%.4f "
                "metric_reg=%.6f mean_abs_c=%.4f mean_complexity=%.4f"
            )
            log_args = [
                epoch,
                step,
                avg_loss,
                avg_ce_loss,
                avg_acc,
                curvature_loss.detach().item(),
                metric_reg_loss.detach().item(),
                mean_abs_curvature,
                mean_complexity,
            ]
            if true_jacobian_enabled and "jacobian_frobenius" in output:
                jacobian_frobenius = output["jacobian_frobenius"].detach()
                jacobian_std = jacobian_frobenius.std(unbiased=False).item()
                corr_complexity_curvature = (
                    safe_correlation(batch["complexity"], output["curvature"].detach().abs())
                    if "complexity" in batch
                    else float("nan")
                )
                corr_complexity_jacobian = (
                    safe_correlation(batch["complexity"], jacobian_frobenius)
                    if "complexity" in batch
                    else float("nan")
                )
                log_message += (
                    " jac_frob_mean=%.4f jac_frob_std=%.4f jac_reg=%.6f "
                    "jac_complexity=%.6f corr_complexity_abs_c=%.4f corr_complexity_jac=%.4f"
                )
                log_args.extend(
                    [
                        jacobian_frobenius.mean().item(),
                        jacobian_std,
                        jacobian_reg_loss.detach().item(),
                        jacobian_complexity_loss.detach().item(),
                        corr_complexity_curvature,
                        corr_complexity_jacobian,
                    ]
                )
                if "jacobian_effective_rank" in output:
                    log_message += " jac_rank_mean=%.4f"
                    log_args.append(torch.nanmean(output["jacobian_effective_rank"].detach()).item())
            logger.info(log_message, *log_args)

    return {
        "loss": total_loss / max(total_count, 1),
        "ce_loss": total_ce_loss / max(total_count, 1),
        "accuracy": total_correct / max(total_count, 1),
    }


def save_jacobian_diagnostics(output: Dict[str, TensorLike], batch: Dict[str, TensorLike], path: Path) -> None:
    """Save a small diagnostic batch; disabled by default."""
    max_items = min(4, int(output["true_jacobian"].size(0)))
    payload = {
        "true_jacobian": output["true_jacobian"][:max_items].detach().cpu(),
        "jacobian_frobenius": output["jacobian_frobenius"][:max_items].detach().cpu(),
        "logits": output["logits"][:max_items].detach().cpu(),
        "labels": batch["labels"][:max_items].detach().cpu(),
    }
    if "query_local_metric" in output:
        payload["query_local_metric"] = output["query_local_metric"][:max_items].detach().cpu()
    if "complexity" in batch:
        payload["complexity"] = batch["complexity"][:max_items].detach().cpu()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


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
    device = get_device()
    logger.info("using device=%s", device)
    if (
        config.get("use_true_jacobian_metric", False)
        and not config.get("use_dummy_encoder", True)
        and config.get("batch_size", 0) > 16
    ):
        logger.warning(
            "true Jacobian metric with a BERT encoder is expensive; batch_size=%d may cause OOM. "
            "Consider a per-GPU batch size of 16 or lower.",
            config["batch_size"],
        )

    tokenizer = build_tokenizer(config)
    train_dataset, val_dataset = build_datasets(config)
    train_loader = build_dataloader(train_dataset, tokenizer, config, shuffle=True, device=device)
    val_loader = build_dataloader(val_dataset, tokenizer, config, shuffle=False, device=device)

    with (output_dir / "config_used.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=True)

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
        use_true_jacobian_metric=config.get("use_true_jacobian_metric", False),
        true_jacobian_target=config.get("true_jacobian_target", "lorentz"),
        true_jacobian_create_graph=config.get("true_jacobian_create_graph", True),
        true_jacobian_normalize=config.get("true_jacobian_normalize", True),
        true_jacobian_identity_mix=config.get("true_jacobian_identity_mix", 0.1),
        true_jacobian_max_batch=config.get("true_jacobian_max_batch"),
        true_jacobian_reg_weight=config.get("true_jacobian_reg_weight", 0.0),
        true_jacobian_complexity_weight=config.get("true_jacobian_complexity_weight", 0.0),
        detach_true_jacobian_metric=config.get("detach_true_jacobian_metric", False),
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
