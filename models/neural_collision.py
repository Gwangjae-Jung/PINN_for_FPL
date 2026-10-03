from   typing import Optional
import torch
from   torch  import nn
from   utils  import FiniteDifferenceMethod


__all__: list[str] = ['CNN_enc_dec', 'generate_operator_D', 'generate_operator_F', 'NeuralCollisionOperator']


##################################################
##################################################
class CNN_enc_dec(nn.Module):
    """Convolutional encoder-decoder neural surrogate for FPL collision sub-operators.

    ## Description
    Encoder-decoder convolutional network that approximates the sub-operators $D(f)$
    and $F(f)$ of the Fokker-Planck-Landau collision operator at a fixed velocity grid
    resolution. Architecture follows opPINN (Lee et al., JCP 2023).

    ## Arguments
    `dim` (`int`): Dimension of velocity space (`2` or `3`).
    `res` (`int`): Velocity grid resolution along each axis.
    `type_of_operator` (`str`): Sub-operator type being approximated (`'D'` or `'F'`).
    `encoder` (`Optional[nn.Sequential]`, default: `None`): Custom encoder network.
    If `None`, a default encoder is constructed via `_get_base_encoder`.
    `decoder` (`Optional[nn.Sequential]`, default: `None`): Custom decoder network.
    If `None`, a default decoder is constructed via `_get_base_decoder`.
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
        """Evaluate the encoder-decoder network on a distribution tensor.

        ## Description
        Passes the input through the encoder, reshapes the bottleneck feature map to
        the expected spatial layout, then passes through the decoder to produce the
        predicted sub-operator output.

        ## Arguments
        `f` (`torch.Tensor`): Input distribution tensor of shape `(batch, 1, *domain)`.

        ## Returns
        `torch.Tensor`: Predicted operator tensor with the same batch size as `f`.
        """
        feat: torch.Tensor = self.encoder.forward(f)
        feat = feat.view(feat.shape[0], 64, *(self.__res // 16 for _ in range(self.__dim)))
        return self.decoder.forward(feat)


##################################################
##################################################
class NeuralCollisionOperator():
    """Neural surrogate collision operator for the Fokker-Planck-Landau equation.

    ## Description
    Evaluates the surrogate FPL collision operator $Q(f)$ using two trained
    CNN encoder-decoder networks (`op_D` approximating $D(f)$ and `op_F` approximating
    $F(f)$) together with a central finite difference divergence computation.

    ## Arguments
    `dimension` (`int`): Dimension of velocity space.
    `resolution` (`int`): Velocity resolution along each axis.
    `v_max` (`float`): Velocity cutoff bound defining `[-v_max, v_max]^d`.
    `op_D` (`torch.nn.Module`): Surrogate network approximating the diffusion operator
    $D(f)$.
    `op_F` (`torch.nn.Module`): Surrogate network approximating the drift operator
    $F(f)$.
    `coeff` (`float`, default: `1.0`): Multiplicative scaling coefficient applied to
    the collision term output.
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
        self.__fdm:      FiniteDifferenceMethod = FiniteDifferenceMethod(
            dimension, 2.0 * v_max / resolution, device=device,
        )
        return

    def forward(self, f: torch.Tensor, grad_f: torch.Tensor) -> torch.Tensor:
        """Evaluate the surrogate collision operator $Q(f)$.

        ## Description
        Uses the surrogate sub-operators $D(f)$ and $F(f)$ together with the supplied
        velocity gradient `grad_f` to compute the divergence form of the Landau
        collision operator via central finite differences.

        ## Arguments
        `f` (`torch.Tensor`): Distribution tensor of shape `(num_points, 1)` arranged
        in `ij`-indexing style over the velocity grid.
        `grad_f` (`torch.Tensor`): Velocity gradient tensor of shape
        `(num_points, dimension)` (excluding the temporal derivative).

        ## Returns
        `torch.Tensor`: Evaluated collision operator tensor matching the shape of `f`.
        """
        old_shape: torch.Size    = f.shape
        dim: int                 = self.__dim
        res: int                 = self.__res
        f_reshaped:      torch.Tensor = f.reshape(self.__shape_tv)
        grad_f_reshaped: torch.Tensor = grad_f.reshape(*(f_reshaped.shape), dim)

        # Evaluate both surrogate sub-operators and reshape to grid layout
        df_pred: torch.Tensor = self.op_D.forward(f_reshaped[:, None])
        ff_pred: torch.Tensor = self.op_F.forward(f_reshaped[:, None])
        df_pred = df_pred.reshape(-1, dim, dim, *(res for _ in range(dim)))
        ff_pred = ff_pred.reshape(-1, dim, *(res for _ in range(dim)))

        # Compute per-axis flux: D(f) * grad_f - F(f) * f
        operands: list[torch.Tensor] = [
            torch.einsum("tj..., t...j -> t...", df_pred[:, d], grad_f_reshaped) - ff_pred[:, d] * f_reshaped
            for d in range(dim)
        ]
        # Compute divergence via finite differences and sum over axes
        diff_operands: list[torch.Tensor] = [
            self.__fdm.compute_derivative(op, idx)
            for idx, op in enumerate(operands)
        ]
        q: torch.Tensor = torch.stack(diff_operands, dim=-1).sum(dim=-1)
        return self.__coeff * q.reshape(old_shape)


##################################################
def _get_base_encoder(dim: int, res: int) -> nn.Sequential:
    """Construct the default convolutional encoder for `CNN_enc_dec`.

    ## Description
    Builds a four-block strided convolutional encoder followed by a linear projection.
    The encoder progressively halves the spatial resolution by a factor of 16 total
    (two stride-2 blocks for 2D and 3D).

    ## Arguments
    `dim` (`int`): Spatial dimension (`2` or `3`).
    `res` (`int`): Input velocity grid resolution along each axis.

    ## Returns
    `nn.Sequential`: Encoder network.
    """
    cfg_encoder: dict[str, int] = {'kernel_size': 5, 'stride': 2, 'padding': 2}
    conv: type = getattr(nn, f'Conv{dim}d')
    if dim in (2, 3):
        return nn.Sequential(
            conv(1,  8,  **cfg_encoder),
            nn.ReLU(),
            conv(8,  16, **cfg_encoder),
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
    """Construct the default convolutional decoder for `CNN_enc_dec`.

    ## Description
    Builds an upsampling transposed-convolutional decoder whose output channel count
    depends on the sub-operator type (`'D'` for the matrix $D(f)$, `'F'` for the
    vector $F(f)$).

    ## Arguments
    `dim` (`int`): Spatial dimension (`2` or `3`).
    `type_of_operator` (`str`): Sub-operator type (`'D'` or `'F'`).

    ## Returns
    `nn.Sequential`: Decoder network.
    """
    cfg_decoder: dict[str, int]    = {'kernel_size': 5, 'padding': 2}
    cfg_upscale: dict[str, object] = {
        'mode':          'bilinear' if dim == 2 else 'trilinear',
        'align_corners': True,
    }
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
            return decoder                      # Output channels: 4 = dim^2 (D is d x d matrix)
        elif type_of_operator == 'F':
            decoder.append(nn.ReLU())
            decoder.append(deconv(4, 2, **cfg_decoder))  # Output channels: 2 = dim (F is a vector)
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
                deconv(16, 9, **cfg_decoder),   # Output channels: 9 = dim^2
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
                deconv(8, 3, **cfg_decoder),    # Output channels: 3 = dim
            )
        raise ValueError(f"Operator type '{type_of_operator}' is not recognized.")
    raise NotImplementedError(f"Dimension {dim} is not supported.")


##################################################
def generate_operator_D(dim: int, res: int, device: Optional[torch.device] = None) -> CNN_enc_dec:
    """Instantiate a surrogate model for the diffusion sub-operator $D(f)$.

    ## Description
    Constructs a `CNN_enc_dec` model configured to approximate the matrix-valued
    diffusion sub-operator $D(f)$ of the Fokker-Planck-Landau collision operator.

    ## Arguments
    `dim` (`int`): Dimension of velocity space.
    `res` (`int`): Velocity grid resolution along each axis.
    `device` (`Optional[torch.device]`, default: `None`): Target computing device.

    ## Returns
    `CNN_enc_dec`: Initialized surrogate model for $D(f)$.
    """
    return CNN_enc_dec(dim, res, type_of_operator='D', device=device)


def generate_operator_F(dim: int, res: int, device: Optional[torch.device] = None) -> CNN_enc_dec:
    """Instantiate a surrogate model for the drift sub-operator $F(f)$.

    ## Description
    Constructs a `CNN_enc_dec` model configured to approximate the vector-valued
    drift sub-operator $F(f)$ of the Fokker-Planck-Landau collision operator.

    ## Arguments
    `dim` (`int`): Dimension of velocity space.
    `res` (`int`): Velocity grid resolution along each axis.
    `device` (`Optional[torch.device]`, default: `None`): Target computing device.

    ## Returns
    `CNN_enc_dec`: Initialized surrogate model for $F(f)$.
    """
    return CNN_enc_dec(dim, res, type_of_operator='F', device=device)


##################################################
##################################################
# End of file