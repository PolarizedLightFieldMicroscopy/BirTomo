import torch
import torch.nn as nn
import torch.nn.functional as F

class VonMisesLoss(nn.Module):
    def __init__(self, kappa=1.0):
        super().__init__()
        self.kappa = kappa

    def forward(self, orientation_pred, orientation_gt):
        diff = orientation_pred - orientation_gt
        loss = 1 - torch.exp(self.kappa * torch.cos(diff))
        return loss.mean()
    


class CosineSimilarityLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, vector_pred, vector_gt):
        cos_sim = F.cosine_similarity(vector_pred, vector_gt, dim=-1)
        loss = 1 - cos_sim
        return loss.mean()


def apply_loss_function_and_reg(loss_type, reg_type, retardance_measurement, orientation_measurement,
                                retardance_estimate, orientation_estimate, ret_orie_weight=0.5,
                                volume_estimate=None, regularization_weight=0.5):
        
        if loss_type=='vonMisses':
            retardance_loss = F.mse_loss(retardance_measurement, retardance_estimate)
            angular_loss = (1 - F.cosine_similarity(orientation_measurement, orientation_estimate)).mean()
            data_term = ret_orie_weight * retardance_loss + (1 - ret_orie_weight) * angular_loss
        
        # Vector difference
        if loss_type=='vector':
            co_gt, ca_gt = retardance_measurement*torch.cos(orientation_measurement), retardance_measurement*torch.sin(orientation_measurement)
            co_pred, ca_pred = retardance_estimate*torch.cos(orientation_estimate), retardance_estimate*torch.sin(orientation_estimate)
            data_term = ((co_gt-co_pred)**2 + (ca_gt-ca_pred)**2).mean()
        
        elif loss_type=='L1_cos':
            data_term = orientation_estimate * (retardance_measurement - retardance_estimate).abs().mean() + \
                (1-ret_orie_weight) * torch.cos(orientation_measurement - orientation_estimate).abs().mean()
            
        elif loss_type=='L1all':
            azimuth_damp_mask = (retardance_measurement / retardance_measurement.max()).detach()
            data_term = (retardance_measurement - retardance_estimate).abs().mean() + \
            (2 * (1 - torch.cos(orientation_measurement - orientation_estimate)) * azimuth_damp_mask).mean()

        if volume_estimate is not None:
            if reg_type=='L1':
            # L1 or sparsity 
                regularization_term = volume_estimate.Delta_n.abs().mean()
            # L2 or sparsity 
            elif reg_type=='L2':
                regularization_term = (volume_estimate.Delta_n**2).mean()
            # Unit length regularizer
            elif reg_type=='unit':
                regularization_term  = (1-(volume_estimate.optic_axis[0,...]**2+volume_estimate.optic_axis[1,...]**2+volume_estimate.optic_axis[2,...]**2)).abs().mean()
            else:
                regularization_term = torch.zeros([1], device=retardance_measurement.device)
        
        L = regularization_weight * data_term + (1-regularization_weight) * regularization_term
        return L, data_term, regularization_term