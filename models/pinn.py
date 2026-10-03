from   typing import Optional
import torch
from   torch  import nn


__all__: list[str] = ['PINN_FPL']


class PINN_FPL(nn.Module):
    """## Physics-informed neural network for the Fokker-Planck-Landau equation

## Description
Multilayer perceptron architecture with Tanh activations and a Softplus output activation
to enforce strict positivity of the predicted distribution function $f(t, v) > 0$.

## Arguments
`dimension` (`int`): Dimension of the velocity domain.
`depth` (`int`): Total number of hidden linear layers.
`width` (`int`): Number of hidden units in each linear layer.
`softplus` (`float`, default: `1.0`): Beta parameter for the Softplus activation function.
`dtype` (`Optional[torch.dtype]`, default: `None`): Tensor data type for model weights.
`device` (`Optional[torch.device]`, default: `None`): Device on which model is instantiated.

## Returns
`None`: None.
"""
    def __init__(
            self,
            dimension: int,
            depth:     int,
            width:     int,
            softplus:  float                  = 1.0,
            dtype:     Optional[torch.dtype]  = None,
            device:    Optional[torch.device] = None,
        ) -> None:
        super().__init__()

        if dtype is None:
            dtype = torch.get_default_dtype()
        if device is None:
            device = torch.get_default_device()

        layers: list[nn.Module] = [
            nn.Linear(1 + dimension, width, dtype=dtype),
            nn.Tanh(),
        ]
        for _ in range(depth - 1):
            layers.append(nn.Linear(width, width, dtype=dtype))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(width, 1, dtype=dtype))
        layers.append(nn.Softplus(beta=softplus))

        self.network: nn.Sequential = nn.Sequential(*layers).to(device)
        return

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """## Forward pass of PINN

## Description
Evaluates the physics-informed neural network at given spatio-temporal coordinate points.

## Arguments
`points` (`torch.Tensor`): Tensor of shape `(num_points, 1 + dimension)` containing `(t, v)` coordinates.

## Returns
`torch.Tensor`: Predicted distribution values of shape `(num_points, 1)`.
"""
        return self.network(points)