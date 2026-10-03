from   typing import Optional
import torch
from   torch  import nn
from   utils  import FiniteDifferenceMethod


__all__: list[str] = ['CNN_enc_dec', 'generate_operator_D', 'generate_operator_F', 'NeuralCollisionOperator']


class CNN_enc_dec(nn.Module):
    """## Convolutional encoder-decoder neural surrogate for collision sub-operators

## Description
Encoder-decoder convolutional network approximating the sub-operators $D(f)$ and $F(f)$
at a fixed velocity grid resolution, following opPINN (Lee et al., JCP 2023).

## Arguments
`dim` (`int`): Dimension of velocity space (`2` or `3`).
`res` (`int`): Velocity grid resolution along each axis.
`type_of_operator` (`str`): Type of operator being approximated (`'D'` or `'F'`).
`encoder` (`Optional[nn.Sequential]`, default: `None`): Custom encoder network.
`decoder` (`Optional[nn.Sequential]`, default: `None`): Custom decoder network.
`device` (`Optional[torch.device]`, default: `None`): Target computing device.

## Returns
`None`: None.
"""
    def __init__(
            self,
            dim:              int,
            res:              int,
            type_of_operator: str,
            encoder:          Optional[nn.Sequential] = None,
            decoder:          Optional[nn.Sequential] = None,
            device:           Optional[torch.device]  = None,
        ) -> None:
        if dim not in (2, 3):
            raise NotImplementedError(f"Only 2D and 3D are supported, got dim={dim}.")
        if type_of_operator not in ('D', 'F'):
            raise ValueError(f"'type_of_operator' must be 'D' or 'F', got '{type_of_operator}'.")

        super().__init__()
        if device is None:
            device = torch.get_default_device()

        self.__dim: int = dim
        self.__res: int = res

        self.encoder: nn.Sequential = encoder if encoder is not None else _get_base_encoder(dim, res)
        self.decoder: nn.Sequential = decoder if decoder is not None else _get_base_decoder(dim, type_of_operator)
        self.to(device)
        return

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        """## Forward evaluation of encoder-decoder network

## Description
Processes input distribution tensor through encoder, reshapes bottleneck representation, and passes through decoder.

## Arguments
`f` (`torch.Tensor`): Input distribution tensor of shape `(batch, 1, *domain)`.

## Returns
`torch.Tensor`: Predicted operator tensor.
"""
        feat: torch.Tensor = self.encoder.forward(f)
        feat = feat.view(feat.shape[0], 64, *(self.__res // 16 for _ in range(self.__dim)))
        return self.decoder.forward(feat)


class NeuralCollisionOperator():
    """## Neural surrogate collision operator evaluator

## Description
Evaluates the surrogate Fokker-Planck-Landau collision operator using trained CNN encoder-decoder models
`op_D` and `op_F` and central finite difference differentiation.

## Arguments
`dimension` (`int`): Dimension of velocity space.
`resolution` (`int`): Velocity resolution along each axis.
`v_max` (`float`): Velocity cutoff bound defining `[-v_max, v_max]^d`.
`op_D` (`torch.nn.Module`): Neural network approximating the diffusion operator $D(f)$.
`op_F` (`torch.nn.Module`): Neural network approximating the drift operator $F(f)$.
`coeff` (`float`, default: `1.0`): Multiplicative coefficient for the collision term.
`device` (`Optional[torch.device]`, default: `None`): Target computing device.

## Returns
`None`: None.
"""
    def __init__(
            self,
            dimension:  int,
            resolution: int,
            v_max:      float,
            op_D:       torch.nn.Module,
            op_F:       torch.nn.Module,
            coeff:      float                  = 1.0,
            device:     Optional[torch.device] = None,
        ) -> None:
        if device is None:
            device = torch.get_default_device()

        self.__dim:      int                    = dimension
        self.__res:      int                    = resolution
        self.op_D:       torch.nn.Module        = op_D.to(device)
        self.op_F:       torch.nn.Module        = op_F.to(device)
        self.op_D.eval()
        self.op_F.eval()
        self.__coeff:    float                  = coeff
        self.__shape_tv: tuple[int, ...]        = tuple([-1, *(resolution for _ in range(dimension))])
        self.__fdm:      FiniteDifferenceMethod = FiniteDifferenceMethod(dimension, 2.0 * v_max / resolution, device=device)
        return

    def forward(self, f: torch.Tensor, grad_f: torch.Tensor) -> torch.Tensor:
        """## Evaluate neural surrogate collision operator

## Description
Computes the collision operator $Q(f)$ using predicted sub-operators $D(f)$ and $F(f)$ and finite difference divergence.

## Arguments
`f` (`torch.Tensor`): Tensor of shape `(num_points, 1)` aligned in `ij`-indexing style.
`grad_f` (`torch.Tensor`): Spatial gradient tensor of shape `(num_points, dimension)` excluding temporal derivative.

## Returns
`torch.Tensor`: Evaluated collision operator tensor matching shape of `f`.
"""
        old_shape: torch.Size = f.shape
        dim: int = self.__dim
        res: int = self.__res
        f_reshaped: torch.Tensor = f.reshape(self.__shape_tv)
        grad_f_reshaped: torch.Tensor = grad_f.reshape(*(f_reshaped.shape), dim)

        df_pred: torch.Tensor = self.op_D.forward(f_reshaped[:, None])
        ff_pred: torch.Tensor = self.op_F.forward(f_reshaped[:, None])
        df_pred = df_pred.reshape(-1, dim, dim, *(res for _ in range(dim)))
        ff_pred = ff_pred.reshape(-1, dim, *(res for _ in range(dim)))

        operands: list[torch.Tensor] = [
            torch.einsum("tj..., t...j -> t...", df_pred[:, d], grad_f_reshaped) - ff_pred[:, d] * f_reshaped
            for d in range(dim)
        ]
        diff_operands: list[torch.Tensor] = [
            self.__fdm.compute_derivative(op, idx)
            for idx, op in enumerate(operands)
        ]
        q: torch.Tensor = torch.stack(diff_operands, dim=-1).sum(dim=-1)
        return self.__coeff * q.reshape(old_shape)


##################################################
def _get_base_encoder(dim: int, res: int) -> nn.Sequential:
    cfg_encoder: dict[str, int] = {'kernel_size': 5, 'stride': 2, 'padding': 2}
    conv: type = getattr(nn, f'Conv{dim}d')
    if dim in (2, 3):
        return nn.Sequential(
            conv(1, 8, **cfg_encoder),
            nn.ReLU(),
            conv(8, 16, **cfg_encoder),
            nn.ReLU(),
            conv(16, 32, **cfg_encoder),
            nn.ReLU(),
            conv(32, 64, **cfg_encoder),
            nn.ReLU(),
            nn.Flatten(start_dim=2),
            nn.Linear((res // 16)**dim, (res // 16)**dim),
        )
    raise NotImplementedError(f"Dimension {dim} is not supported.")


def _get_base_decoder(dim: int, type_of_operator: str) -> nn.Sequential:
    cfg_decoder: dict[str, int] = {'kernel_size': 5, 'padding': 2}
    cfg_upscale: dict[str, object] = {'mode': 'bilinear' if dim == 2 else 'trilinear', 'align_corners': True}
    deconv: type = getattr(nn, f'ConvTranspose{dim}d')
    if dim == 2:
        decoder: nn.Sequential = nn.Sequential(
            deconv(64, 32, **cfg_decoder),
            nn.ReLU(),
            nn.Upsample(scale_factor=4, **cfg_upscale),
            deconv(32, 16, **cfg_decoder),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, **cfg_upscale),
            deconv(16, 8, **cfg_decoder),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, **cfg_upscale),
            deconv(8, 4, **cfg_decoder),
        )
        if type_of_operator == 'D':
            return decoder
        elif type_of_operator == 'F':
            decoder.append(nn.ReLU())
            decoder.append(deconv(4, 2, **cfg_decoder))
            return decoder
        raise ValueError(f"Operator type '{type_of_operator}' is not recognized.")
    elif dim == 3:
        if type_of_operator == 'D':
            return nn.Sequential(
                deconv(64, 32, **cfg_decoder),
                nn.ReLU(),
                nn.Upsample(scale_factor=4, **cfg_upscale),
                deconv(32, 16, **cfg_decoder),
                nn.ReLU(),
                nn.Upsample(scale_factor=4, **cfg_upscale),
                deconv(16, 9, **cfg_decoder),
            )
        elif type_of_operator == 'F':
            return nn.Sequential(
                deconv(64, 32, **cfg_decoder),
                nn.ReLU(),
                nn.Upsample(scale_factor=4, **cfg_upscale),
                deconv(32, 16, **cfg_decoder),
                nn.ReLU(),
                nn.Upsample(scale_factor=2, **cfg_upscale),
                deconv(16, 8, **cfg_decoder),
                nn.ReLU(),
                nn.Upsample(scale_factor=2, **cfg_upscale),
                deconv(8, 3, **cfg_decoder),
            )
        raise ValueError(f"Operator type '{type_of_operator}' is not recognized.")
    raise NotImplementedError(f"Dimension {dim} is not supported.")


def generate_operator_D(dim: int, res: int, device: Optional[torch.device] = None) -> CNN_enc_dec:
    """## Instantiate D-operator surrogate model

## Description
Instantiates a `CNN_enc_dec` model configured to approximate the diffusion sub-operator $D(f)$.

## Arguments
`dim` (`int`): Dimension of velocity space.
`res` (`int`): Velocity grid resolution.
`device` (`Optional[torch.device]`, default: `None`): Target computing device.

## Returns
`CNN_enc_dec`: Initialized surrogate model for $D(f)$.
"""
    return CNN_enc_dec(dim, res, type_of_operator='D', device=device)


def generate_operator_F(dim: int, res: int, device: Optional[torch.device] = None) -> CNN_enc_dec:
    """## Instantiate F-operator surrogate model

## Description
Instantiates a `CNN_enc_dec` model configured to approximate the drift sub-operator $F(f)$.

## Arguments
`dim` (`int`): Dimension of velocity space.
`res` (`int`): Velocity grid resolution.
`device` (`Optional[torch.device]`, default: `None`): Target computing device.

## Returns
`CNN_enc_dec`: Initialized surrogate model for $F(f)$.
"""
    return CNN_enc_dec(dim, res, type_of_operator='F', device=device)