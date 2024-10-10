import torch
import torch.nn.functional as F
from VolumeRaytraceLFM.birefringence_implementations import BirefringentVolume


def compare_volumes(volume1: BirefringentVolume, volume2: BirefringentVolume, mask=None, only_nonzero=False):
    if mask is not None:
        Delta_n1 = volume1.get_delta_n().detach()
        Delta_n2 = volume2.get_delta_n().detach()
        Delta_n1 = Delta_n1[mask]
        Delta_n2 = Delta_n2[mask]
    else:
        Delta_n1 = volume1.get_delta_n().detach()
        Delta_n2 = volume2.get_delta_n().detach()
    optic_axis1 = volume1.get_optic_axis().detach()
    optic_axis2 = volume2.get_optic_axis().detach()
    
    if only_nonzero:
        non_zero_mask = (Delta_n1 != 0) | (Delta_n2 != 0)
        Delta_n1 = Delta_n1[non_zero_mask]
        Delta_n2 = Delta_n2[non_zero_mask]
        optic_axis1 = optic_axis1[:, non_zero_mask]
        optic_axis2 = optic_axis2[:, non_zero_mask]
    
    # Combine Delta_n and optic_axis into a single tensor for both volumes
    # Stack Delta_n and optic_axis such that Delta_n corresponds to index [0, ...]
    predicted_volume = torch.cat((Delta_n1.unsqueeze(0), optic_axis1), dim=0)  # (4, H, W, D)
    target_volume = torch.cat((Delta_n2.unsqueeze(0), optic_axis2), dim=0)     # (4, H, W, D)

    # Compute the loss
    loss = mse_sum(predicted_volume, target_volume)

    return loss


def mse_sum(predicted_volume, target_volume):
    """Compute the birefringence field loss"""
    vector_field1 = predicted_volume[0, ...] * predicted_volume[1:, ...]
    vector_field2 = target_volume[0, ...] * target_volume[1:, ...]
    loss = F.mse_loss(vector_field1, vector_field2, reduction='sum')
    return loss
