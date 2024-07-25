import json
import torch
import torch.nn.functional as F
from VolumeRaytraceLFM.metrics.data_fidelity import (
    poisson_loss,
    gaussian_noise_loss,
    complex_mse_loss,
)
from VolumeRaytraceLFM.metrics.regularization import (
    l2_bir,
    l2_bir_active,
    l1_bir,
    total_variation_bir,
    total_variation_bir_subset,
    cosine_similarity_neighbors,
    neg_penalty_bir_active,
    pos_penalty_bir_active,
    pos_penalty_l2_bir_active,
)


REGULARIZATION_FCNS = {
    "birefringence L2": l2_bir,
    "birefringence active L2": l2_bir_active,
    "birefringence L1": l1_bir,
    "birefringence TV": total_variation_bir,
    "birefringence active TV": total_variation_bir_subset,
    "local cosine similarity": cosine_similarity_neighbors,
    "birefringence active negative penalty": neg_penalty_bir_active,
    "birefringence active positive penalty": pos_penalty_bir_active,
    "birefringence active positive penalty L2": pos_penalty_l2_bir_active,
}


class PolarimetricLossFunction:
    def __init__(self, params=None, json_file=None):
        if params or json_file:
            if json_file:
                with open(json_file, "r") as f:
                    params = json.load(f)
            self.weight_retardance = params.get("weight_retardance", 1.0)
            self.weight_orientation = params.get("weight_orientation", 1.0)
            self.weight_datafidelity = params.get("weight_datafidelity", 1.0)
            self.weight_regularization = params.get("regularization_weight", 1.0)
            # Initialize specific loss functions
            self.optimizer = params.get("optimizer", "Adam")
            self.datafidelity = params.get("datafidelity", "vector")
            self.regularization_fcns = [
                (REGULARIZATION_FCNS[fn_name], weight)
                for fn_name, weight in params.get("regularization_fcns", [])
            ]
        else:
            self.weight_retardance = 1.0
            self.weight_orientation = 1.0
            self.weight_datafidelity = 1.0
            self.weight_regularization = 0.1
            self.optimizer = "Adam"
            self.datafidelity = "vector"
            self.regularization_fcns = []

    def set_retardance_target(self, target):
        self.target_retardance = target

    def set_orientation_target(self, target):
        self.target_orientation = target

    def set_intensity_list_target(self, target_list):
        self.target_intensity_list = target_list

    def compute_retardance_loss(self, prediction):
        # Add logic to transform data and compute retardance loss
        pass

    def compute_orientation_loss(self, prediction):
        # Add logic to transform data and compute orientation loss
        pass

    def transform_ret_azim_to_vector_form(self, ret, azim):
        """Transform the retardance (ret) and azimuth (azim) into vector form.
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

    def transform_ret_azim_to_euler_form(self, ret, azim):
        """Transform the retardance (ret) and azimuth (azim) into Euler form.
        Args:
        - ret (torch.Tensor): A tensor containing the retardance image.
        - azim (torch.Tensor): A tensor containing the azimuth image.
        Returns:
        - (torch.Tensor): A tensors representing Euler's formula
        """
        ret = ret.to(torch.cfloat)
        azim = azim.to(torch.cfloat)
        euler = ret * torch.exp(2 * 1j * azim)
        return euler

    def vector_loss(self, ret_pred, azim_pred):
        """Compute the vector loss"""
        ret_gt = self.target_retardance
        azim_gt = self.target_orientation
        cos_gt, sin_gt = self.transform_ret_azim_to_vector_form(ret_gt, azim_gt)
        cos_pred, sin_pred = self.transform_ret_azim_to_vector_form(ret_pred, azim_pred)
        loss_cos = F.mse_loss(cos_pred, cos_gt)
        loss_sin = F.mse_loss(sin_pred, sin_gt)
        loss = loss_cos + loss_sin
        return loss

    def euler_loss(self, ret_pred, azim_pred):
        """Compute the vector loss"""
        ret_gt = self.target_retardance
        azim_gt = self.target_orientation
        euler_gt = self.transform_ret_azim_to_euler_form(ret_gt, azim_gt)
        euler_pred = self.transform_ret_azim_to_euler_form(ret_pred, azim_pred)
        loss = complex_mse_loss(euler_gt, euler_pred)
        return loss

    def intensity_loss(self, intensity_list_pred):
        """Compute the intensity loss"""
        intensity_list_gt = self.target_intensity_list
        losses = [
            F.mse_loss(pred, gt)
            for pred, gt in zip(intensity_list_pred, intensity_list_gt)
        ]
        total_loss = torch.mean(torch.stack(losses)) * 10
        return total_loss

    def compute_datafidelity_term(self, method, *args):
        """Compares the predicted data with the target data.
        Args:
        method (str): The method to use, defaults to 'vector'.
        *args: Depending on `method`:
            - If 'vector', expects (ret_pred, azim_pred).
            - If 'intensity', expects (intensity_list_pred).
        """
        first_word = method.split()[0]
        second_word = method.split()[1] if len(method.split()) > 1 else None
        if first_word == "vector":
            ret_pred, azim_pred = args[0]
            data_loss = self.vector_loss(ret_pred, azim_pred)
        elif first_word == "euler":
            ret_pred, azim_pred = args[0]
            data_loss = self.euler_loss(ret_pred, azim_pred)
        elif first_word == "intensity":
            intensity_list_gt = self.target_intensity_list
            intensity_list_pred = args[0]
            if second_word == "mse" or second_word is None:
                losses = [
                    F.mse_loss(pred, gt)
                    for pred, gt in zip(intensity_list_pred, intensity_list_gt)
                ]
                data_loss = torch.mean(torch.stack(losses)) * 10
            elif second_word == "poisson":
                losses = [
                    poisson_loss(pred, gt)
                    for pred, gt in zip(intensity_list_pred, intensity_list_gt)
                ]
                data_loss = torch.mean(torch.stack(losses))
            elif second_word == "gaussian":
                losses = [
                    gaussian_noise_loss(pred, gt, sigma=0.4)
                    for pred, gt in zip(intensity_list_pred, intensity_list_gt)
                ]
                data_loss = torch.mean(torch.stack(losses)) * 0.05
            else:
                raise ValueError(f"Invalid intensity method: {method}")
        elif first_word == "retazim":
            retardance_loss = self.compute_retardance_loss(args[0])
            orientation_loss = self.compute_orientation_loss(args[1])
            data_loss = (
                self.weight_retardance * retardance_loss
                + self.weight_regularization * orientation_loss
            )
        else:
            raise ValueError(f"Invalid data fidelity method: {method}")
        return data_loss * 1000

    def reg_l2(self, data):
        return l2_bir(data)

    def reg_l1(self, data):
        return l1_bir(data)

    def reg_tv(self, data):
        return total_variation_bir(data)

    def reg_cosine_similarity(self, data):
        return cosine_similarity_neighbors(data)

    def compute_regularization_term(self, data):
        """Compute regularization term"""
        if not self.regularization_fcns:
            return torch.tensor(0.0, device=data.device), []

        term_values = []

        # Start with the first regularization function directly
        first_reg_fn, first_weight = self.regularization_fcns[0]
        first_term_value = first_weight * first_reg_fn(data) * 1000
        term_values.append(first_term_value)
        regularization_loss = first_term_value.clone()

        # Sum up the rest of the regularization terms if any
        for reg_fcn, weight in self.regularization_fcns[1:]:
            term_value = weight * reg_fcn(data) * 1000
            term_values.append(term_value)
            regularization_loss += term_value

        return regularization_loss, term_values

    def compute_total_loss(self, ret_pred, azim_pred, data):
        # Compute individual losses
        datafidelity_loss = self.compute_datafidelity_term(ret_pred, azim_pred)
        regularization_loss = self.compute_regularization_term(data)

        # Compute total loss with weighted sum
        total_loss = (
            self.weight_datafidelity * datafidelity_loss
            + self.weight_regularization * regularization_loss
        )
        return total_loss


class RetAzimLoss(torch.nn.Module):
    def __init__(self):
        super(RetAzimLoss, self).__init__()

    def forward(self, predicted_images, images):
        """Compute the vector loss"""
        euler_pred = torch.polar(predicted_images[:, 0], 2 * predicted_images[:, 1])
        euler_gt = torch.polar(images[:, 0], 2 * images[:, 1])
        abs_MSE_loss = torch.mean(torch.abs(euler_pred - euler_gt) ** 2)
        return abs_MSE_loss
