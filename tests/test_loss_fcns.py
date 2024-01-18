import torch
from VolumeRaytraceLFM.loss_functions import weighted_local_cosine_similarity_loss


def test_weighted_local_cosine_similarity_loss():
    """Test that the local cosine similarity loss is a scalar
    between 0 and 2.
    """
    for _ in range(10):
        optic_axis = torch.randn(3, 10, 10, 10)  
        delta_n = torch.randn(10, 10, 10)

        loss = weighted_local_cosine_similarity_loss(optic_axis, delta_n)

        assert loss.ndim == 0, "Loss is not a scalar value."
        assert 0 <= loss.item() <= 2, f"Loss {loss:.6f} is not within the range [0, 2]."
