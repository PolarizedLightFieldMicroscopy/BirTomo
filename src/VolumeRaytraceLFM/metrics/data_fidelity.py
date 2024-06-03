'''Data-fidelity metrics'''
import torch
import torch.nn.functional as F


def poisson_loss(y_pred, y_true):
    """Poisson loss function.
    Args:
    y_pred (torch.Tensor): Predicted rates (lambda), must be non-negative.
    y_true (torch.Tensor): Observed counts (y).
    
    Returns:
    torch.Tensor: Computed Poisson loss.
    """
    # Adding a small value to prevent log(0)
    loss = y_pred - y_true * torch.log(y_pred + 1e-8)
    return torch.mean(loss)


def gaussian_noise_loss(predictions, targets, sigma=1.0):
    """
    Gaussian Noise Loss function based on the negative log likelihood
    of a Gaussian distribution.

    Args:
    predictions (torch.Tensor): The predictions of the model.
    targets (torch.Tensor): The true values.
    sigma (float): The standard deviation of the Gaussian noise.

    Returns:
    torch.Tensor: The computed Gaussian noise loss.
    """
    sigma = torch.tensor(sigma, dtype=predictions.dtype, device=predictions.device)
    residual = predictions - targets
    # Compute the negative log likelihood of the residual under Gaussian noise assumption
    loss = 0.5 * torch.log(2 * torch.pi * sigma ** 2) + (residual ** 2) / (2 * sigma ** 2)
    return torch.mean(loss)


def von_mises_loss(angle_pred, angle_gt, kappa=1.0):
    '''Von Mises loss function for orientation'''
    diff = angle_pred - angle_gt
    loss = 1 - torch.exp(kappa * torch.cos(diff))
    return loss.mean()


def cosine_similarity_loss(vector_pred, vector_gt):
    '''Cosine similarity loss function for orientation'''
    cos_sim = F.cosine_similarity(vector_pred, vector_gt, dim=-1)
    loss = 1 - cos_sim
    return loss.mean()


def complex_mse_loss(output, target):
    # Compute the element-wise difference and
    #   then the squared magnitude of the difference
    diff = output - target
    squared_magnitude = torch.abs(diff)**2
    mse_loss = torch.mean(squared_magnitude)
    return mse_loss
