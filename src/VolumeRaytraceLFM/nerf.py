"""This script defines a PyTorch-based Implicit Neural Representation (INR) using a 
Multi-Layer Perceptron (MLP) with custom sine activations and weight initialization. 
The INR represents a continuous function mapping input coordinates to output properties.

Classes:
- Sine: Custom sine activation function for use in the INR.
- ImplicitRepresentationMLP: MLP that acts as an implicit neural representation.

Functions:
- sine_init: Custom weight initialization for sine activations.
- setup_optimizer_nerf: Sets up the optimizer for the neural network model.
- generate_voxel_grid: Generates a grid of voxel coordinates for a given volume shape.
- predict_voxel_properties: Predicts properties for each voxel in the grid using the given model.
- get_model_device: Returns the device of the parameters of the model.

Example usage:
- Initialize the ImplicitRepresentationMLP with specified input and output dimensions.
- Generate voxel coordinates and predict properties using the model.
"""

import torch
import torch.nn as nn
import math


class Sine(nn.Module):
    def forward(self, x):
        return torch.sin(x)


def sine_init(m):
    with torch.no_grad():
        if isinstance(m, nn.Linear):
            num_input = m.weight.size(-1)
            std = math.sqrt(6 / num_input)
            m.weight.uniform_(-std, std)
            if m.bias is not None:
                m.bias.uniform_(-std, std)


class ImplicitRepresentationMLP(nn.Module):
    """Multi-Layer Perceptron (MLP) that acts as an
    Implicit Neural Representation.

    Args:
        input_dim (int): Dimensionality of the input.
        output_dim (int): Dimensionality of the output.
        hidden_layers (list): List of integers defining the number of
            neurons in each hidden layer.
        num_frequencies (int): Number of frequencies for positional encoding.
    """

    def __init__(
        self, input_dim, output_dim, hidden_layers=[128, 64], num_frequencies=10
    ):
        super(ImplicitRepresentationMLP, self).__init__()
        self.num_frequencies = num_frequencies
        self.input_dim = input_dim
        self.output_dim = output_dim

        layers = []
        in_dim = input_dim * (2 * num_frequencies + 1)
        for h in hidden_layers:
            layers.append(nn.Linear(in_dim, h))
            layers.append(Sine())  # Using Sine activation for INR
            in_dim = h
        layers.append(nn.Linear(in_dim, output_dim))
        self.layers = nn.Sequential(*layers)
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights of the network using custom sine initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                sine_init(m)  # Sine initialization for hidden layers
        self._initialize_output_layer()

    def _initialize_output_layer(self):
        """Initialize the weights of the output layer."""
        final_layer = self.layers[-1]
        with torch.no_grad():
            final_layer.weight.uniform_(-0.01, 0.01)
            final_layer.bias[0] = 0.05  # First output dimension fixed to 0.05
            final_layer.bias[1:].uniform_(-0.5, 0.5)  # Initializing other biases

    def positional_encoding(self, x: torch.Tensor) -> torch.Tensor:
        """Apply positional encoding to the input tensor.
        Each element of x is multiplied by each frequency, effectively
        encoding the input in a higher-dimensional space.
        Args:
            x (torch.Tensor): Input tensor of shape (N, input_dim).
        Returns:
            torch.Tensor: Encoded tensor of shape (N, input_dim * (2 * num_frequencies + 1)).
        """
        frequencies = torch.linspace(
            0, self.num_frequencies - 1, self.num_frequencies, device=x.device
        )
        frequencies = 2.0**frequencies
        x_expanded = x.unsqueeze(-1) * frequencies.unsqueeze(0).unsqueeze(0)
        x_sin = torch.sin(x_expanded)
        x_cos = torch.cos(x_expanded)
        x_encoded = torch.cat([x.unsqueeze(-1), x_sin, x_cos], dim=-1)
        return x_encoded.view(x.size(0), -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.
        Args:
            x (torch.Tensor): Input tensor of shape (N, input_dim).
        Returns:
            torch.Tensor: Output tensor of shape (N, output_dim).
        """
        x = self.positional_encoding(x)
        x = self.layers(x)
        # Scaling the outputs
        x[:, 0] = torch.sigmoid(x[:, 0]) * 0.1  # First output dimension around 0.05
        x[:, 1] = torch.sigmoid(x[:, 1])  # Second output dimension between 0 and 1
        x[:, 2:4] = torch.tanh(x[:, 2:4])  # Last two dimensions between -1 and 1
        return x


class ImplicitRepresentationMLPSpherical(nn.Module):
    """Multi-Layer Perceptron (MLP) that acts as an
    Implicit Neural Representation.

    Args:
        input_dim (int): Dimensionality of the input.
        output_dim (int): Dimensionality of the output.
        hidden_layers (list): List of integers defining the number of
            neurons in each hidden layer.
        num_frequencies (int): Number of frequencies for positional encoding.
    """

    def __init__(
        self, input_dim, output_dim, params_dict=None
    ):
        super(ImplicitRepresentationMLPSpherical, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.params_dict = params_dict
        hidden_layers = self.params_dict.get("hidden_layers", [128, 64])
        num_frequencies = self.params_dict.get("num_frequencies", 10)
        self.num_frequencies = num_frequencies

        layers = []
        in_dim = input_dim * (2 * num_frequencies + 1)
        for h in hidden_layers:
            layers.append(nn.Linear(in_dim, h))
            layers.append(Sine())  # Using Sine activation for INR
            in_dim = h
        layers.append(nn.Linear(in_dim, output_dim))
        self.layers = nn.Sequential(*layers)
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights of the network using custom sine initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                sine_init(m)  # Sine initialization for hidden layers
        self._initialize_output_layer()

    def _initialize_output_layer(self):
        """Initialize the weights of the output layer."""
        final_layer = self.layers[-1]
        weight_range = self.params_dict.get("final_layer_weight_range", [-0.01, 0.01])
        birefringence_bias = self.params_dict.get("final_layer_bias_birefringence", 0.05)
        with torch.no_grad():

            final_layer.weight.uniform_(weight_range[0], weight_range[1])
            final_layer.bias[0] = birefringence_bias
            final_layer.bias[1:].uniform_(-0.5, 0.5)

    def positional_encoding(self, x: torch.Tensor) -> torch.Tensor:
        """Apply positional encoding to the input tensor.
        Each element of x is multiplied by each frequency, effectively
        encoding the input in a higher-dimensional space.
        Args:
            x (torch.Tensor): Input tensor of shape (N, input_dim).
        Returns:
            torch.Tensor: Encoded tensor of shape (N, input_dim * (2 * num_frequencies + 1)).
        """
        frequencies = 2.0**torch.arange(0, self.num_frequencies, device=x.device)
        x_expanded = x.unsqueeze(-1) * frequencies.unsqueeze(0).unsqueeze(0)
        x_sin = torch.sin(x_expanded)
        x_cos = torch.cos(x_expanded)
        x_encoded = torch.cat([x.unsqueeze(-1), x_sin, x_cos], dim=-1)
        return x_encoded.view(x.size(0), -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.
        Args:
            x (torch.Tensor): Input tensor of shape (N, input_dim).
        Returns:
            torch.Tensor: Output tensor of shape (N, output_dim).
        """
        x = self.positional_encoding(x)
        x = self.layers(x)
        # x[:, 1] = x[:, 1] % (2 * torch.pi)  # Second output dimension between 0 and 2pi
        # x[:, 2] = x[:, 2] % (torch.pi / 2)  # Third output dimension between 0 and pi/2
        # x_new[:, 1] = torch.atan2(torch.sin(x[:, 1]), torch.cos(x[:, 1]))  # Azimuthal angle (phi) between -pi and pi
        # x_new[:, 2] = torch.acos(torch.clamp(x[:, 2], -1.0, 1.0))  # Polar angle (theta) between 0 and pi
    
        # Scaling and constraining the outputs
        x_new = x.clone()   # Clone the tensor to avoid in-place operations
        x_new[:, 0] = torch.sigmoid(x[:, 0]) * 0.1  # Density output in [0, 0.1]
        x_new[:, 1] = torch.remainder(x[:, 1], 2 * torch.pi)  # Angle in [0, 2π]
        x_new[:, 2] = torch.remainder(x[:, 2], torch.pi / 2) # Angle in [0, π/2]
        x = x_new
        return x


def setup_optimizer_nerf(
    model: nn.Module, training_params: dict
) -> torch.optim.Optimizer:
    """Set up the optimizer for the neural network model with layer-specific learning rates.

    Args:
        model (nn.Module): The neural network model.
        training_params (dict): Dictionary containing training parameters such as learning rate.
    Returns:
        torch.optim.Optimizer: Configured optimizer.
    """
    if isinstance(model, nn.DataParallel):
        model = model.module  # Access the actual model inside DataParallel

    # Extract the NeRF-specific parameters
    nerf_params = training_params.get("nerf", {})

    # Extract NeRF learning rates
    lr_fc1 = nerf_params.get("learning_rates", {}).get("fc1", 1e-2)  # Learning rate for fc1
    lr_fc2 = nerf_params.get("learning_rates", {}).get("fc2", 1e-4)  # Learning rate for fc2
    lr_fc3 = nerf_params.get("learning_rates", {}).get("fc3", 1e-4)  # Learning rate for fc3
    lr_output = nerf_params.get("learning_rates", {}).get("output", 1e-4)  # Learning rate for output layer

    # Extract optimizer parameters from the JSON
    optimizer_type = nerf_params.get("optimizer", {}).get("type", "NAdam")
    betas = tuple(nerf_params.get("optimizer", {}).get("betas", [0.9, 0.999]))  # Tuple for betas
    eps = nerf_params.get("optimizer", {}).get("eps", 1e-8)
    weight_decay = nerf_params.get("optimizer", {}).get("weight_decay", 1e-4)

    # Access layers from model (assuming it's an instance of ImplicitRepresentationMLPSpherical)
    parameters = [
        # fc1 layer
        {
            "params": model.layers[0].parameters(),  # First Linear layer (fc1)
            "lr": lr_fc1,
        },
        # fc2 layer
        {
            "params": model.layers[2].parameters(),  # Second Linear layer (fc2)
            "lr": lr_fc2,
        },
        # fc3 layer
        {
            "params": model.layers[4].parameters(),  # Third Linear layer (fc3)
            "lr": lr_fc3,
        },
        # Output layer
        {
            "params": model.layers[-1].parameters(),  # Output Linear layer
            "lr": lr_output,
        },
    ]
    # Setup the optimizer using the NAdam parameters from the JSON
    optimizer = torch.optim.NAdam(
        parameters,
        betas=betas,           # Momentum coefficients from the JSON
        eps=eps,               # Epsilon for numerical stability
        weight_decay=weight_decay,  # Weight decay for regularization
    )
    return optimizer


def generate_voxel_grid(vol_shape: tuple) -> torch.Tensor:
    """Generate a grid of voxel coordinates for a given volume shape.
    Args:
        vol_shape (tuple): Shape of the volume (D, H, W).
    Returns:
        torch.Tensor: Tensor of shape (D*H*W, 3) containing voxel
                      coordinates.
    """
    x = torch.linspace(0, vol_shape[0] - 1, vol_shape[0])
    y = torch.linspace(0, vol_shape[1] - 1, vol_shape[1])
    z = torch.linspace(0, vol_shape[2] - 1, vol_shape[2])
    grid = torch.meshgrid(x, y, z, indexing="ij")
    coords = torch.stack(grid, dim=-1).reshape(-1, 3)
    return coords


def predict_voxel_properties(model: nn.Module, vol_shape: tuple, enable_grad=False):
    """Predict properties for each voxel in the grid using the given model.
    Args:
        model (nn.Module): The neural network model.
        vol_shape (tuple): Shape of the volume (D, H, W).
    Returns:
        torch.Tensor: Predicted properties reshaped to the
                      volume shape (D, H, W, C).
    """
    device = get_model_device(model)
    coords = generate_voxel_grid(vol_shape).float().to(device)
    vol_shape_tensor = torch.tensor(vol_shape, dtype=coords.dtype, device=device)
    coords_normalized = coords / vol_shape_tensor  # Normalize coordinates if necessary
    if enable_grad:
        coords_normalized.requires_grad_(True)
        output = model(coords_normalized)
    else:
        with torch.no_grad():
            output = model(coords_normalized)  # .cpu()
    return output.reshape(*vol_shape, -1)


def get_model_device(model: nn.Module):
    """Returns the device of the parameters of the model."""
    return next(model.parameters()).device
