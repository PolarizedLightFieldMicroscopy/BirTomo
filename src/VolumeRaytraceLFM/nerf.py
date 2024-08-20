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
        self, input_dim, output_dim, hidden_layers=[128, 64], num_frequencies=10
    ):
        super(ImplicitRepresentationMLPSpherical, self).__init__()
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
        # # Scaling the outputs
        # x[:, 0] = torch.sigmoid(x[:, 0]) * 0.1  # First output dimension around 0.05
        # x[:, 1] = x[:, 1] % (2 * torch.pi)  # Second output dimension between 0 and 2pi
        # x[:, 2] = x[:, 2] % (torch.pi / 2)  # Third output dimension between 0 and pi/2
        # x_new = x.clone()  # Clone the tensor to avoid in-place operations
        # x_new[:, 1] = torch.atan2(torch.sin(x[:, 1]), torch.cos(x[:, 1]))  # Azimuthal angle (phi) between -pi and pi
        # x_new[:, 2] = torch.acos(torch.clamp(x[:, 2], -1.0, 1.0))  # Polar angle (theta) between 0 and pi
        # x = x_new  # Assign the modified tensor back to x
        return x


def setup_optimizer_nerf(
    model: nn.Module, training_params: dict
) -> torch.optim.Optimizer:
    """Set up the optimizer for the neural network model.
    TODO: use the training_params to set the optimizer parameters.

    Args:
        model (nn.Module): The neural network model.
        training_params (dict): Dictionary containing training parameters such as learning rate.
    Returns:
        torch.optim.Optimizer: Configured optimizer.
    """
    inr_params = model.inr_model.parameters()
    parameters = [
        {
            "params": inr_params,
            "lr": 0.01,
            # "lr": training_params.get("lr", 0.001),
        }
    ]
    # optimizer_class = getattr(torch.optim, training_params.get("optimizer", "NAdam"))
    # optimizer = optimizer_class(parameters)
    optimizer = torch.optim.NAdam(parameters)  # , lr=0.001)
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


def predict_voxel_properties(model: nn.Module, vol_shape: tuple):
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
    with torch.no_grad():
        output = model(coords_normalized)  # .cpu()
    return output.reshape(*vol_shape, -1)


def get_model_device(model: nn.Module):
    """Returns the device of the parameters of the model."""
    return next(model.parameters()).device
