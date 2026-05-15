import torch

from src.distance import LorentzBilinearDistance, lorentz_expmap0


def test_identity_query_local_metric_matches_original_pairwise_distance() -> None:
    torch.manual_seed(0)
    batch_size = 3
    num_labels = 4
    tangent_dim = 5
    coordinate_dim = tangent_dim + 1

    c_x = -torch.ones(batch_size, 1)
    c_y = -torch.ones(num_labels, 1)
    x = lorentz_expmap0(0.05 * torch.randn(batch_size, tangent_dim), c_x)
    y = lorentz_expmap0(0.05 * torch.randn(num_labels, tangent_dim), c_y)
    distance = LorentzBilinearDistance(coordinate_dim=coordinate_dim)

    identity_metric = torch.eye(coordinate_dim).expand(batch_size, -1, -1)
    original = distance.pairwise(x, y, c_x, c_y)
    corrected = distance.pairwise(x, y, c_x, c_y, query_local_metric=identity_metric)

    assert corrected.shape == (batch_size, num_labels)
    assert torch.allclose(corrected, original, atol=1e-6, rtol=1e-6)
