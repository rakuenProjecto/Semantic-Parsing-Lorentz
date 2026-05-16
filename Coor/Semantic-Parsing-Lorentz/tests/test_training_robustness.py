import logging
import math

import torch

from src.dataset import MockSemanticParsingDataset, SimpleWhitespaceTokenizer
from src.model import SemanticLorentzParser
from train import (
    aggregate_validation_diagnostics,
    build_dataloader,
    curvature_anti_collapse_losses,
    normalize_config_types,
    safe_pearson_correlation,
    target_abs_curvature_from_complexity,
)


def test_config_type_normalization_handles_scientific_strings_and_booleans() -> None:
    config = {
        "learning_rate": "2e-5",
        "weight_decay": "0.01",
        "batch_size": "2",
        "epochs": "3",
        "hidden_dim": "32",
        "tangent_dim": "8",
        "use_true_jacobian_metric": "false",
        "true_jacobian_create_graph": "true",
    }

    normalize_config_types(config)

    assert config["learning_rate"] == 2e-5
    assert isinstance(config["learning_rate"], float)
    assert config["use_true_jacobian_metric"] is False
    assert config["true_jacobian_create_graph"] is True
    assert config["batch_size"] == 2
    assert isinstance(config["batch_size"], int)


def test_safe_pearson_correlation_validity_flags() -> None:
    too_small = safe_pearson_correlation([1.0], [2.0])
    constant = safe_pearson_correlation([1.0, 1.0, 1.0], [0.0, 1.0, 2.0])
    normal = safe_pearson_correlation([1.0, 2.0, 3.0], [1.0, 3.0, 5.0])

    assert too_small["valid"] is False
    assert too_small["reason"] == "too_few_samples"
    assert constant["valid"] is False
    assert constant["reason"] == "near_zero_variance"
    assert normal["valid"] is True
    assert normal["value"] is not None
    assert math.isfinite(normal["value"])


def test_validation_diagnostics_json_contains_expected_keys(tmp_path) -> None:
    torch.manual_seed(0)
    device = torch.device("cpu")
    config = {
        "output_dir": str(tmp_path),
        "batch_size": 2,
        "num_workers": 0,
        "max_length": 16,
        "min_abs_curvature": 0.05,
        "max_abs_curvature": 5.0,
        "curvature_target_mode": "complexity_linear",
        "curvature_min_warning_fraction": 0.8,
        "use_true_jacobian_metric": False,
        "enable_curvature_anti_collapse": True,
    }
    normalize_config_types(config)

    dataset = MockSemanticParsingDataset(
        num_samples=4,
        num_labels=3,
        seed=0,
        complexity_mode="heuristic",
    )
    tokenizer = SimpleWhitespaceTokenizer(vocab_size=256)
    dataloader = build_dataloader(dataset, tokenizer, config, shuffle=False, device=device)
    model = SemanticLorentzParser(
        num_labels=3,
        use_dummy_encoder=True,
        dummy_vocab_size=256,
        hidden_dim=16,
        tangent_dim=4,
        max_position_embeddings=16,
        dropout=0.0,
    ).to(device)

    diagnostics = aggregate_validation_diagnostics(
        model=model,
        dataloader=dataloader,
        device=device,
        config=config,
        logger=logging.getLogger("test_validation_diagnostics"),
        epoch=1,
    )

    expected_keys = {
        "mean_abs_c",
        "std_abs_c",
        "mean_complexity",
        "jac_frob_mean",
        "jac_frob_std",
        "jac_effective_rank_mean",
        "corr_complexity_abs_c",
        "corr_complexity_abs_c_valid",
        "corr_complexity_jac",
        "corr_complexity_jac_valid",
        "curvature_min_fraction",
        "curvature_max_fraction",
        "curvature_mean_to_min_ratio",
        "curvature_aux_loss",
        "target_abs_curvature_mean",
        "pred_abs_c_min",
        "pred_abs_c_max",
        "pred_abs_c_mean",
        "pred_abs_c_std",
        "target_abs_c_min",
        "target_abs_c_max",
        "target_abs_c_mean",
        "target_abs_c_std",
        "curvature_target_scale_used",
        "curvature_collapse_penalty",
        "curvature_spread_penalty",
        "curvature_total_aux_loss",
        "curvature_anti_collapse_weight",
        "curvature_spread_weight",
        "curvature_init_abs_c",
    }
    assert expected_keys.issubset(diagnostics.keys())
    assert (tmp_path / "diagnostics_epoch_1.json").exists()


def test_curvature_anti_collapse_penalty_is_finite_scalar() -> None:
    config = {
        "min_abs_curvature": 0.05,
        "max_abs_curvature": 5.0,
        "curvature_target_mode": "complexity_linear",
        "enable_curvature_anti_collapse": True,
        "curvature_min_margin": 0.1,
        "curvature_min_std": 0.02,
        "curvature_raw_floor_penalty_weight": 0.05,
    }
    normalize_config_types(config)
    pred_abs_c = torch.tensor([[0.0501], [0.0502], [0.0503]])
    complexity = torch.tensor([[1.0], [2.0], [4.0]])
    target_abs_c = target_abs_curvature_from_complexity(complexity, config, epoch=1, pred_abs_curvature=pred_abs_c)

    collapse_penalty, spread_penalty, distribution_penalty = curvature_anti_collapse_losses(
        pred_abs_c,
        target_abs_c,
        config,
    )

    for penalty in (collapse_penalty, spread_penalty, distribution_penalty):
        assert penalty.shape == torch.Size([])
        assert torch.isfinite(penalty)
