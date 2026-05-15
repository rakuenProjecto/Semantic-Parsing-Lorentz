"""Train the dynamic-curvature Lorentz semantic parser."""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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


def as_bool(value: Any) -> bool:
    """Parse booleans from config or CLI values."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in {0, 1}:
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y", "on"}:
            return True
        if normalized in {"false", "0", "no", "n", "off"}:
            return False
    raise ValueError(f"cannot parse boolean value: {value!r}")


def as_int(value: Any) -> int:
    """Parse integer config values, accepting numeric strings."""
    if value is None:
        raise ValueError("cannot parse None as int")
    if isinstance(value, bool):
        raise ValueError(f"cannot parse boolean as int: {value!r}")
    return int(value)


def as_float(value: Any) -> float:
    """Parse float config values, accepting scientific notation strings."""
    if value is None:
        raise ValueError("cannot parse None as float")
    if isinstance(value, bool):
        raise ValueError(f"cannot parse boolean as float: {value!r}")
    return float(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Lorentz semantic parser")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config")

    parser.add_argument("--seed", type=int)
    parser.add_argument("--use_dummy_encoder", dest="use_dummy_encoder", nargs="?", const=True, type=as_bool, default=None)
    parser.add_argument("--no_dummy_encoder", dest="use_dummy_encoder", action="store_false")
    parser.add_argument("--encoder_name", type=str)
    parser.add_argument("--hf_local_files_only", dest="hf_local_files_only", nargs="?", const=True, type=as_bool, default=None)
    parser.add_argument("--no_hf_local_files_only", dest="hf_local_files_only", action="store_false")
    parser.add_argument("--freeze_encoder", dest="freeze_encoder", nargs="?", const=True, type=as_bool, default=None)
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
    parser.add_argument(
        "--return_complexity_features",
        dest="return_complexity_features",
        nargs="?",
        const=True,
        type=as_bool,
        default=None,
    )
    parser.add_argument("--no_return_complexity_features", dest="return_complexity_features", action="store_false")

    parser.add_argument(
        "--use_true_jacobian_metric",
        dest="use_true_jacobian_metric",
        nargs="?",
        const=True,
        type=as_bool,
        default=None,
    )
    parser.add_argument("--no_true_jacobian_metric", dest="use_true_jacobian_metric", action="store_false")
    parser.add_argument("--true_jacobian_target", choices=["lorentz", "tangent"])
    parser.add_argument(
        "--true_jacobian_create_graph",
        dest="true_jacobian_create_graph",
        nargs="?",
        const=True,
        type=as_bool,
        default=None,
    )
    parser.add_argument("--no_true_jacobian_create_graph", dest="true_jacobian_create_graph", action="store_false")
    parser.add_argument(
        "--true_jacobian_normalize",
        dest="true_jacobian_normalize",
        nargs="?",
        const=True,
        type=as_bool,
        default=None,
    )
    parser.add_argument("--no_true_jacobian_normalize", dest="true_jacobian_normalize", action="store_false")
    parser.add_argument("--true_jacobian_identity_mix", type=float)
    parser.add_argument("--true_jacobian_max_batch", type=int)
    parser.add_argument("--true_jacobian_reg_weight", type=float)
    parser.add_argument("--true_jacobian_complexity_weight", type=float)
    parser.add_argument(
        "--detach_true_jacobian_metric",
        dest="detach_true_jacobian_metric",
        nargs="?",
        const=True,
        type=as_bool,
        default=None,
    )
    parser.add_argument("--no_detach_true_jacobian_metric", dest="detach_true_jacobian_metric", action="store_false")
    parser.add_argument(
        "--log_true_jacobian_stats",
        dest="log_true_jacobian_stats",
        nargs="?",
        const=True,
        type=as_bool,
        default=None,
    )
    parser.add_argument("--no_log_true_jacobian_stats", dest="log_true_jacobian_stats", action="store_false")
    parser.add_argument("--true_jacobian_log_interval", type=int)
    parser.add_argument(
        "--save_jacobian_diagnostics",
        dest="save_jacobian_diagnostics",
        nargs="?",
        const=True,
        type=as_bool,
        default=None,
    )
    parser.add_argument("--no_save_jacobian_diagnostics", dest="save_jacobian_diagnostics", action="store_false")
    parser.add_argument("--curvature_min_warning_fraction", type=float)
    parser.add_argument("--curvature_target_mode", choices=["complexity_linear", "complexity_squared", "none"])

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
    normalize_config_types(config)
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
        "curvature_min_warning_fraction": 0.8,
        "curvature_target_mode": "complexity_linear",
    }
    for key, value in defaults.items():
        config.setdefault(key, value)


def normalize_config_types(config: Dict[str, Any]) -> None:
    """Coerce config values after YAML loading and CLI overrides.

    Some YAML parsers or hand-edited config files may treat values such as
    ``2e-5`` as strings. Explicit normalization keeps optimizer construction
    and numeric comparisons robust across environments.
    """
    apply_config_defaults(config)
    bool_fields = {
        "use_dummy_encoder",
        "hf_local_files_only",
        "freeze_encoder",
        "return_complexity_features",
        "use_true_jacobian_metric",
        "true_jacobian_create_graph",
        "true_jacobian_normalize",
        "detach_true_jacobian_metric",
        "log_true_jacobian_stats",
        "save_jacobian_diagnostics",
    }
    int_fields = {
        "seed",
        "num_labels",
        "hidden_dim",
        "tangent_dim",
        "dummy_vocab_size",
        "max_position_embeddings",
        "max_length",
        "num_train_samples",
        "num_val_samples",
        "batch_size",
        "num_workers",
        "epochs",
        "log_interval",
        "true_jacobian_log_interval",
    }
    optional_int_fields = {"true_jacobian_max_batch"}
    float_fields = {
        "learning_rate",
        "weight_decay",
        "dropout",
        "min_abs_curvature",
        "max_abs_curvature",
        "max_expmap_norm",
        "max_tangent_norm",
        "metric_delta_scale",
        "metric_reg_weight",
        "curvature_aux_weight",
        "true_jacobian_reg_weight",
        "true_jacobian_complexity_weight",
        "true_jacobian_identity_mix",
        "grad_clip_norm",
        "curvature_min_warning_fraction",
    }

    for field in bool_fields:
        if field in config and config[field] is not None:
            config[field] = as_bool(config[field])
    for field in int_fields:
        if field in config and config[field] is not None:
            config[field] = as_int(config[field])
    for field in optional_int_fields:
        if field in config and config[field] is not None:
            config[field] = as_int(config[field])
    for field in float_fields:
        if field in config and config[field] is not None:
            config[field] = as_float(config[field])

    if config["curvature_target_mode"] not in {"complexity_linear", "complexity_squared", "none"}:
        raise ValueError("curvature_target_mode must be one of: complexity_linear, complexity_squared, none")
    if not 0.0 <= config["curvature_min_warning_fraction"] <= 1.0:
        raise ValueError("curvature_min_warning_fraction must be in [0, 1]")


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


def safe_pearson_correlation(
    x: TensorLike,
    y: TensorLike,
    eps: float = 1e-8,
    min_samples: int = 3,
) -> Dict[str, Any]:
    """Return Pearson correlation plus a validity flag.

    Small batches and near-constant vectors produce misleading values such as
    exactly +/-1 or NaN. Callers should log ``valid`` alongside ``value``.
    """
    x_tensor = torch.as_tensor(x).detach().float().view(-1)
    y_tensor = torch.as_tensor(y).detach().float().view(-1)
    count = min(x_tensor.numel(), y_tensor.numel())
    if count < min_samples:
        return {"value": None, "valid": False, "reason": "too_few_samples", "n": int(count)}
    x_tensor = x_tensor[:count]
    y_tensor = y_tensor[:count]
    x_centered = x_tensor - x_tensor.mean()
    y_centered = y_tensor - y_tensor.mean()
    denom = x_centered.norm() * y_centered.norm()
    if denom.item() <= eps:
        return {"value": None, "valid": False, "reason": "near_zero_variance", "n": int(count)}
    value = float((x_centered * y_centered).sum().div(denom).item())
    if not math.isfinite(value):
        return {"value": None, "valid": False, "reason": "non_finite", "n": int(count)}
    return {"value": value, "valid": True, "reason": "", "n": int(count)}


def format_optional_float(value: Optional[float]) -> float:
    return float("nan") if value is None else float(value)


def normalized_complexity_target(complexity: torch.Tensor, mode: str) -> Optional[torch.Tensor]:
    """Map complexity in [1, 4] to a normalized curvature target in [0, 1]."""
    if mode == "none":
        return None
    normalized = ((complexity - 1.0) / 3.0).clamp(0.0, 1.0)
    if mode == "complexity_linear":
        return normalized
    if mode == "complexity_squared":
        return normalized.pow(2)
    raise ValueError("curvature_target_mode must be one of: complexity_linear, complexity_squared, none")


def target_abs_curvature_from_complexity(complexity: torch.Tensor, config: Dict[str, Any]) -> Optional[torch.Tensor]:
    normalized_target = normalized_complexity_target(complexity, config.get("curvature_target_mode", "complexity_linear"))
    if normalized_target is None:
        return None
    curvature_span = config["max_abs_curvature"] - config["min_abs_curvature"]
    return config["min_abs_curvature"] + curvature_span * normalized_target


def curvature_boundary_tolerance(config: Dict[str, Any]) -> float:
    return max(1e-4, 0.1 * float(config["min_abs_curvature"]))


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
            target_abs_curvature_mean = torch.full((), float("nan"), dtype=loss.dtype, device=loss.device)

            if is_train and config["metric_reg_weight"] > 0:
                loss = loss + config["metric_reg_weight"] * metric_reg_loss
            if (
                is_train
                and config.get("curvature_aux_weight", 0.0) > 0
                and "complexity" in batch
            ):
                complexity = batch["complexity"].view(-1, 1)
                target_abs_curvature = target_abs_curvature_from_complexity(complexity, config)
                if target_abs_curvature is not None:
                    target_abs_curvature_mean = target_abs_curvature.mean()
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
                "metric_reg=%.6f mean_abs_c=%.4f mean_complexity=%.4f target_abs_c_mean=%.4f"
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
                target_abs_curvature_mean.detach().item(),
            ]
            if true_jacobian_enabled and "jacobian_frobenius" in output:
                jacobian_frobenius = output["jacobian_frobenius"].detach()
                jacobian_std = jacobian_frobenius.std(unbiased=False).item()
                corr_complexity_curvature = (
                    safe_pearson_correlation(batch["complexity"], output["curvature"].detach().abs())
                    if "complexity" in batch
                    else {"value": None, "valid": False, "reason": "missing_complexity", "n": 0}
                )
                corr_complexity_jacobian = (
                    safe_pearson_correlation(batch["complexity"], jacobian_frobenius)
                    if "complexity" in batch
                    else {"value": None, "valid": False, "reason": "missing_complexity", "n": 0}
                )
                log_message += (
                    " jac_frob_mean=%.4f jac_frob_std=%.4f jac_reg=%.6f "
                    "jac_complexity=%.6f corr_complexity_abs_c_batch=%.4f "
                    "corr_complexity_abs_c_batch_valid=%s corr_complexity_jac_batch=%.4f "
                    "corr_complexity_jac_batch_valid=%s"
                )
                log_args.extend(
                    [
                        jacobian_frobenius.mean().item(),
                        jacobian_std,
                        jacobian_reg_loss.detach().item(),
                        jacobian_complexity_loss.detach().item(),
                        format_optional_float(corr_complexity_curvature["value"]),
                        str(corr_complexity_curvature["valid"]).lower(),
                        format_optional_float(corr_complexity_jacobian["value"]),
                        str(corr_complexity_jacobian["valid"]).lower(),
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


def aggregate_validation_diagnostics(
    model: SemanticLorentzParser,
    dataloader: DataLoader,
    device: torch.device,
    config: Dict[str, Any],
    logger: logging.Logger,
    epoch: int,
) -> Dict[str, Any]:
    """Aggregate curvature/Jacobian diagnostics across the full validation set."""
    was_training = model.training
    model.eval()

    true_jacobian_enabled = bool(config.get("use_true_jacobian_metric", False))
    abs_curvatures: List[torch.Tensor] = []
    complexities: List[torch.Tensor] = []
    jacobian_frobenius_values: List[torch.Tensor] = []
    jacobian_rank_values: List[torch.Tensor] = []
    target_abs_curvatures: List[torch.Tensor] = []
    curvature_aux_losses: List[torch.Tensor] = []

    with torch.set_grad_enabled(true_jacobian_enabled):
        for batch in dataloader:
            batch = move_batch_to_device(batch, device)
            output = model(
                input_ids=batch["input_ids"],
                attention_mask=batch.get("attention_mask"),
                labels=batch["labels"],
                return_diagnostics=true_jacobian_enabled,
            )
            abs_curvature = output["curvature"].detach().abs().view(-1)
            abs_curvatures.append(abs_curvature.cpu())
            if "complexity" in batch:
                complexity = batch["complexity"].detach().view(-1)
                complexities.append(complexity.cpu())
                target_abs_curvature = target_abs_curvature_from_complexity(complexity.view(-1, 1), config)
                if target_abs_curvature is not None:
                    target_abs_curvatures.append(target_abs_curvature.detach().view(-1).cpu())
                    per_sample_loss = F.mse_loss(
                        abs_curvature.view(-1, 1),
                        target_abs_curvature.detach(),
                        reduction="none",
                    ).view(-1)
                    curvature_aux_losses.append(per_sample_loss.cpu())
            if "jacobian_frobenius" in output:
                jacobian_frobenius_values.append(output["jacobian_frobenius"].detach().view(-1).cpu())
            if "jacobian_effective_rank" in output:
                jacobian_rank_values.append(output["jacobian_effective_rank"].detach().view(-1).cpu())

    if was_training:
        model.train(True)

    abs_curvature_all = torch.cat(abs_curvatures) if abs_curvatures else torch.empty(0)
    complexity_all = torch.cat(complexities) if complexities else torch.empty(0)
    jacobian_frobenius_all = (
        torch.cat(jacobian_frobenius_values) if jacobian_frobenius_values else torch.empty(0)
    )
    jacobian_rank_all = torch.cat(jacobian_rank_values) if jacobian_rank_values else torch.empty(0)
    target_abs_curvature_all = (
        torch.cat(target_abs_curvatures) if target_abs_curvatures else torch.empty(0)
    )
    curvature_aux_loss_all = torch.cat(curvature_aux_losses) if curvature_aux_losses else torch.empty(0)

    min_abs_curvature = float(config["min_abs_curvature"])
    max_abs_curvature = float(config["max_abs_curvature"])
    boundary_tol = curvature_boundary_tolerance(config)
    min_fraction = (
        float((abs_curvature_all <= min_abs_curvature + boundary_tol).float().mean().item())
        if abs_curvature_all.numel() > 0
        else None
    )
    max_fraction = (
        float((abs_curvature_all >= max_abs_curvature - boundary_tol).float().mean().item())
        if abs_curvature_all.numel() > 0
        else None
    )
    corr_complexity_abs_c = (
        safe_pearson_correlation(complexity_all, abs_curvature_all)
        if complexity_all.numel() > 0
        else {"value": None, "valid": False, "reason": "missing_complexity", "n": 0}
    )
    corr_complexity_jac = (
        safe_pearson_correlation(complexity_all, jacobian_frobenius_all)
        if complexity_all.numel() > 0 and jacobian_frobenius_all.numel() > 0
        else {"value": None, "valid": False, "reason": "missing_jacobian", "n": 0}
    )

    diagnostics = {
        "epoch": epoch,
        "num_samples": int(abs_curvature_all.numel()),
        "mean_abs_c": tensor_mean(abs_curvature_all),
        "std_abs_c": tensor_std(abs_curvature_all),
        "mean_complexity": tensor_mean(complexity_all),
        "jac_frob_mean": tensor_mean(jacobian_frobenius_all),
        "jac_frob_std": tensor_std(jacobian_frobenius_all),
        "jac_effective_rank_mean": tensor_mean(jacobian_rank_all),
        "corr_complexity_abs_c": corr_complexity_abs_c["value"],
        "corr_complexity_abs_c_valid": corr_complexity_abs_c["valid"],
        "corr_complexity_abs_c_reason": corr_complexity_abs_c["reason"],
        "corr_complexity_jac": corr_complexity_jac["value"],
        "corr_complexity_jac_valid": corr_complexity_jac["valid"],
        "corr_complexity_jac_reason": corr_complexity_jac["reason"],
        "curvature_min_fraction": min_fraction,
        "curvature_max_fraction": max_fraction,
        "curvature_mean_to_min_ratio": (
            tensor_mean(abs_curvature_all) / min_abs_curvature
            if tensor_mean(abs_curvature_all) is not None and min_abs_curvature > 0
            else None
        ),
        "curvature_aux_loss": tensor_mean(curvature_aux_loss_all),
        "target_abs_curvature_mean": tensor_mean(target_abs_curvature_all),
        "curvature_boundary_tolerance": boundary_tol,
    }

    write_json(Path(config["output_dir"]) / f"diagnostics_epoch_{epoch}.json", diagnostics)
    logger.info(
        "validation_diagnostics epoch=%d mean_abs_c=%.4f std_abs_c=%.4f mean_complexity=%.4f "
        "jac_frob_mean=%.4f jac_frob_std=%.4f corr_complexity_abs_c_val=%.4f "
        "corr_complexity_abs_c_val_valid=%s corr_complexity_jac_val=%.4f "
        "corr_complexity_jac_val_valid=%s curvature_min_fraction=%.4f "
        "curvature_max_fraction=%.4f curvature_mean_to_min_ratio=%.4f "
        "curvature_aux_loss=%.6f target_abs_curvature_mean=%.4f",
        epoch,
        format_optional_float(diagnostics["mean_abs_c"]),
        format_optional_float(diagnostics["std_abs_c"]),
        format_optional_float(diagnostics["mean_complexity"]),
        format_optional_float(diagnostics["jac_frob_mean"]),
        format_optional_float(diagnostics["jac_frob_std"]),
        format_optional_float(diagnostics["corr_complexity_abs_c"]),
        str(diagnostics["corr_complexity_abs_c_valid"]).lower(),
        format_optional_float(diagnostics["corr_complexity_jac"]),
        str(diagnostics["corr_complexity_jac_valid"]).lower(),
        format_optional_float(diagnostics["curvature_min_fraction"]),
        format_optional_float(diagnostics["curvature_max_fraction"]),
        format_optional_float(diagnostics["curvature_mean_to_min_ratio"]),
        format_optional_float(diagnostics["curvature_aux_loss"]),
        format_optional_float(diagnostics["target_abs_curvature_mean"]),
    )
    warning_fraction = float(config.get("curvature_min_warning_fraction", 0.8))
    if diagnostics["curvature_min_fraction"] is not None and diagnostics["curvature_min_fraction"] > warning_fraction:
        logger.warning(
            "curvature collapse monitor: %.1f%% of validation samples are near min_abs_curvature=%.6f",
            100.0 * diagnostics["curvature_min_fraction"],
            min_abs_curvature,
        )
    return diagnostics


def tensor_mean(values: torch.Tensor) -> Optional[float]:
    if values.numel() == 0:
        return None
    value = float(values.float().mean().item())
    return value if math.isfinite(value) else None


def tensor_std(values: torch.Tensor) -> Optional[float]:
    if values.numel() == 0:
        return None
    value = float(values.float().std(unbiased=False).item())
    return value if math.isfinite(value) else None


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(json_safe(payload), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


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
    final_diagnostics: Optional[Dict[str, Any]] = None
    logger.info("labels=%s", train_dataset.label_names)

    for epoch in range(1, config["epochs"] + 1):
        train_metrics = run_epoch(model, train_loader, optimizer, device, config, logger, epoch)
        val_metrics = run_epoch(model, val_loader, None, device, config, logger, epoch)
        final_diagnostics = aggregate_validation_diagnostics(model, val_loader, device, config, logger, epoch)
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

    if final_diagnostics is not None:
        write_json(output_dir / "diagnostics_final.json", final_diagnostics)
    logger.info("training complete best_val_loss=%.4f checkpoint=%s", best_val_loss, best_path)


if __name__ == "__main__":
    main()
