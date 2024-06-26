"""Utility functions for optimizers."""

from tqdm import tqdm


def calculate_adjusted_lr(optimizer):
    """
    Calculate adjusted learning rates for each parameter managed by the optimizer.

    Args:
        optimizer (torch.optim.Optimizer): The optimizer to calculate rates for.

    Returns:
        dict: A dictionary mapping parameter names to their adjusted learning rates.
    """
    adjusted_lrs = {}
    for group in optimizer.param_groups:
        base_lr = group["lr"]
        for p in group["params"]:
            state = optimizer.state[p]
            if "step" in state and state["step"] > 0:
                beta1, beta2 = group["betas"]
                step = state["step"]
                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step

                # Retrieve moment estimates
                m = state["exp_avg"]
                v = state["exp_avg_sq"]

                # Bias-corrected moment estimates
                m_hat = m / bias_correction1
                v_hat = v / bias_correction2

                # Effective learning rate considering the moment estimates
                adjusted_lr = base_lr * (v_hat.sqrt() + group["eps"]) / m_hat
                adjusted_lrs[id(p)] = adjusted_lr.abs().mean()
            else:
                adjusted_lrs[id(p)] = base_lr  # Default to base lr if not adjusted yet
    return adjusted_lrs


def print_lrs(model, adjusted_lrs):
    """Print the learning rates for each parameter in the model."""
    param_dict = {id(p): name for name, p in model.named_parameters()}
    for p_id, adjusted_lr in adjusted_lrs.items():
        print(
            f"Parameter: {param_dict.get(p_id, 'Unnamed Parameter')} - Adjusted LR: {adjusted_lr}"
        )


def print_adjusted_learning_rates(optimizer):
    for group in optimizer.param_groups:
        base_lr = group["lr"]
        for p in group["params"]:
            state = optimizer.state[p]
            if "step" in state and state["step"] > 0:
                # Extracting beta1, beta2 and epsilon from the optimizer
                beta1, beta2 = group["betas"]
                eps = group["eps"]

                # The formula for the adjusted learning rate in Adam-style optimizers
                # lr_t = lr * sqrt(1 - beta2^t) / (1 - beta1^t)
                # where t is the number of steps taken.
                step = state["step"]
                exp_avg_sq = state["exp_avg_sq"]
                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step
                adjusted_lr = base_lr * (bias_correction2**0.5) / bias_correction1

                print(f"Param ID: {id(p)} - Adjusted LR: {adjusted_lr}")
            else:
                print(f"Param ID: {id(p)} - Adjusted LR: Not yet adjusted")


def print_moments(optimizer):
    # Print the first moment (m) and second moment (v) for each parameter
    for group in optimizer.param_groups:
        for p in group["params"]:
            param_state = optimizer.state[p]
            if "exp_avg" in param_state and "exp_avg_sq" in param_state:
                try:
                    tqdm.write(f"Parameter group: {group['name']}")
                except:
                    tqdm.write(f"Parameter ID: {id(p)}")
                exp_avg = param_state["exp_avg"]
                exp_avg_sq = param_state["exp_avg_sq"]
                exp_avg_beg = exp_avg[:, :5] if p.dim() > 1 else exp_avg[:5]
                exp_avg_sq_beg = exp_avg_sq[:, :5] if p.dim() > 1 else exp_avg_sq[:5]
                tqdm.write(f"exp_avg (m) [at most 5 values]: {exp_avg_beg}")
                tqdm.write(f"exp_avg_sq (v) [at most 5 values]: {exp_avg_sq_beg}")
