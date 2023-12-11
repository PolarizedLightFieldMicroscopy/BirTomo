'''Data-fidelity metrics'''
import torch
import torch.nn.functional as F

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
