import torch

from src.model import SemanticLorentzParser


def test_tiny_model_true_jacobian_forward_has_finite_outputs() -> None:
    torch.manual_seed(0)
    batch_size = 2
    num_labels = 3
    model = SemanticLorentzParser(
        num_labels=num_labels,
        use_dummy_encoder=True,
        dummy_vocab_size=256,
        hidden_dim=32,
        tangent_dim=8,
        max_position_embeddings=16,
        dropout=0.0,
        use_true_jacobian_metric=True,
        true_jacobian_target="lorentz",
        true_jacobian_create_graph=True,
    )
    model.eval()

    input_ids = torch.randint(103, 200, (batch_size, 8))
    input_ids[:, 0] = 101
    attention_mask = torch.ones_like(input_ids)

    output = model(input_ids=input_ids, attention_mask=attention_mask, return_diagnostics=True)

    assert output["logits"].shape == (batch_size, num_labels)
    assert output["jacobian_frobenius"].shape == (batch_size,)
    assert torch.isfinite(output["logits"]).all()
    assert torch.isfinite(output["jacobian_frobenius"]).all()


def test_curvature_parameterization_outputs_positive_finite_abs_curvature() -> None:
    torch.manual_seed(0)
    input_ids = torch.randint(103, 200, (3, 8))
    input_ids[:, 0] = 101
    attention_mask = torch.ones_like(input_ids)

    for parameterization in ("sigmoid_bounded", "softplus_floor"):
        model = SemanticLorentzParser(
            num_labels=3,
            use_dummy_encoder=True,
            dummy_vocab_size=256,
            hidden_dim=32,
            tangent_dim=8,
            max_position_embeddings=16,
            dropout=0.0,
            min_abs_curvature=0.05,
            max_abs_curvature=5.0,
            curvature_parameterization=parameterization,
        )
        output = model(input_ids=input_ids, attention_mask=attention_mask)
        abs_curvature = output["curvature"].abs()

        assert torch.isfinite(abs_curvature).all()
        assert torch.all(abs_curvature > 0.0)
        assert torch.all(abs_curvature >= 0.05)
