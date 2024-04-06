import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from VolumeRaytraceLFM.metrics.regularization import (
    l2_bir,
    l1_bir,
    total_variation_bir,
    cosine_similarity_neighbors,
)


REGULARIZATION_FNS = {
    'birefringence L2': l2_bir,
    'birefringence L1': l1_bir,
    'birefringence TV': total_variation_bir,
    'local cosine similarity': cosine_similarity_neighbors,
}


class PolarimetricLossFunction:
    def __init__(self, params=None, json_file=None):
        if params or json_file:
            if json_file:
                with open(json_file, 'r') as f:
                    params = json.load(f)
            self.weight_retardance = params.get('weight_retardance', 1.0)
            self.weight_orientation = params.get('weight_orientation', 1.0)
            self.weight_datafidelity = params.get('weight_datafidelity', 1.0)
            self.weight_regularization = params.get(
                'regularization_weight', 1.0)
            # Initialize specific loss functions
            self.mse_loss = nn.MSELoss()
            self.regularization_fcns = [(REGULARIZATION_FNS[fn_name], weight)
                                       for fn_name, weight in params.get('regularization_fcns', [])]
        else:
            self.weight_retardance = 1.0
            self.weight_orientation = 1.0
            self.weight_datafidelity = 1.0
            self.weight_regularization = 0.1
            self.mse_loss = nn.MSELoss()
            self.regularization_fcns = []

    def set_retardance_target(self, target):
        self.target_retardance = target

    def set_orientation_target(self, target):
        self.target_orientation = target

    def compute_retardance_loss(self, prediction):
        # Add logic to transform data and compute retardance loss
        pass

    def compute_orientation_loss(self, prediction):
        # Add logic to transform data and compute orientation loss
        pass

    def transform_ret_azim_to_vector_form(self, ret, azim):
        """ Transform the retardance (ret) and azimuth (azim) into vector form.
        Args:
        - ret (torch.Tensor): A tensor containing the retardance image.
        - azim (torch.Tensor): A tensor containing the azimuth image.
        Returns:
        - (torch.Tensor, torch.Tensor): Two tensors representing the
                        cosine and sine components of the vector form.
        """
        # Calculate the cosine and sine components
        cosine_term = ret * torch.cos(2 * azim)
        sine_term = ret * torch.sin(2 * azim)
        return cosine_term, sine_term

    def vector_loss(self, ret_pred, azim_pred):
        '''Compute the vector loss'''
        ret_gt = self.target_retardance
        azim_gt = self.target_orientation
        cos_gt, sin_gt = self.transform_ret_azim_to_vector_form(ret_gt, azim_gt)
        cos_pred, sin_pred = self.transform_ret_azim_to_vector_form(ret_pred, azim_pred)
        loss_cos = F.mse_loss(cos_pred, cos_gt)
        loss_sin = F.mse_loss(sin_pred, sin_gt)
        loss = loss_cos + loss_sin
        return loss

    def compute_datafidelity_term(self, ret_pred, azim_pred, method='vector'):
        '''Incorporates the retardance and orientation losses'''
        if method == 'vector':
            data_loss = self.vector_loss(ret_pred, azim_pred)
        else:
            retardance_loss = self.compute_retardance_loss(ret_pred)
            orientation_loss = self.compute_orientation_loss(azim_pred)
            data_loss = (self.weight_retardance * retardance_loss +
                        self.weight_regularization * orientation_loss)
        return data_loss

    def reg_l2(self, data):
        return l2_bir(data)

    def reg_l1(self, data):
        return l1_bir(data)

    def reg_tv(self, data):
        return total_variation_bir(data)

    def reg_cosine_similarity(self, data):
        return cosine_similarity_neighbors(data)

    def compute_regularization_term(self, data):
        '''Compute regularization term'''
        if not self.regularization_fcns:
            return torch.tensor(0., device=data.device)

        # Start with the first regularization function directly
        first_reg_fn, first_weight = self.regularization_fcns[0]
        regularization_loss = first_weight * first_reg_fn(data)

        # Sum up the rest of the regularization terms if any
        for reg_fn, weight in self.regularization_fcns[1:]:
            regularization_loss += weight * reg_fn(data)

        return regularization_loss

    def compute_total_loss(self, pred_retardance, pred_orientation, data):
        # Compute individual losses
        datafidelity_loss = self.compute_datafidelity_term(
            pred_retardance, pred_orientation)
        regularization_loss = self.compute_regularization_term(data)

        # Compute total loss with weighted sum
        total_loss = (self.weight_datafidelity * datafidelity_loss +
                      self.weight_regularization * regularization_loss)
        return total_loss
