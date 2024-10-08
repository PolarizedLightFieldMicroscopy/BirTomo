"""Functions for monitoring the gradients of the neural network layers."""
import torch


def monitor_gradients(model):
    print("Monitoring layer gradients:")
    
    if isinstance(model, torch.nn.DataParallel):
        model = model.module

    # First hidden layer (fc1)
    if model.layers[0].weight.grad is not None:
        print(f"Layer 1 (fc1) weight gradient norm: {model.layers[0].weight.grad.norm(2).item():.4f}")
    if model.layers[0].bias.grad is not None:
        print(f"Layer 1 (fc1) bias gradient norm: {model.layers[0].bias.grad.norm(2).item():.4f}")

    # Second hidden layer (fc2)
    if model.layers[2].weight.grad is not None:
        print(f"Layer 2 (fc2) weight gradient norm: {model.layers[2].weight.grad.norm(2).item():.4f}")
    if model.layers[2].bias.grad is not None:
        print(f"Layer 2 (fc2) bias gradient norm: {model.layers[2].bias.grad.norm(2).item():.4f}")

    # Third hidden layer (fc3)
    if model.layers[4].weight.grad is not None:
        print(f"Layer 3 (fc3) weight gradient norm: {model.layers[4].weight.grad.norm(2).item():.4f}")
    if model.layers[4].bias.grad is not None:
        print(f"Layer 3 (fc3) bias gradient norm: {model.layers[4].bias.grad.norm(2).item():.4f}")

    # Output layer
    if model.layers[-1].weight.grad is not None:
        print(f"Output layer weight gradient norm: {model.layers[-1].weight.grad.norm(2).item():.4f}")
    if model.layers[-1].bias.grad is not None:
        print(f"Output layer bias gradient norm: {model.layers[-1].bias.grad.norm(2).item():.4f}")


def clip_gradient_norms_nerf(model, iteration_num, verbose=False):
    # Gradient clipping
    max_norm = 1.0
    total_norm = torch.nn.utils.clip_grad_norm_(model.module.parameters(), max_norm=max_norm)
    
    if verbose:
        print(f"Iteration {iteration_num}: Total gradient norm: {total_norm:.2f}")
        if total_norm > max_norm:
            print(f"Iteration {iteration_num}: Gradients clipped to norm {max_norm}")
