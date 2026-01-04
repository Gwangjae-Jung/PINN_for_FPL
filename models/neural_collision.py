from    typing      import  Self, Optional
import  torch
from    torch       import  nn


__all__: list[str] = ['CNN_enc_dec', 'generate_operator_D', 'generate_operator_F', 'NeuralCollisionOperator']


##################################################
##################################################
def _get_base_encoder(dim: int, res: int) -> nn.Sequential:
    cfg_encoder = {'kernel_size': 5, 'stride': 2, 'padding': 2}
    conv = getattr(nn, f'Conv{dim}d')
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
            nn.Linear((res//16)**dim, (res//16)**dim),
        )
    else:
        raise NotImplementedError(f"Dimension {dim} is not supported.")


def _get_base_decoder(dim: int, type_of_operator: str) -> nn.Sequential:
    cfg_decoder = {'kernel_size': 5, 'padding': 2}
    cfg_upscale = {'mode': 'bilinear' if dim==2 else 'trilinear', 'align_corners': True}
    deconv = getattr(nn, f'ConvTranspose{dim}d')
    if dim==2:
        decoder = nn.Sequential(
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
        if type_of_operator=='D':
            return decoder
        elif type_of_operator=='F':
            decoder.append(nn.ReLU())
            decoder.append(deconv(4, 2, **cfg_decoder))
            return decoder
        else:
            raise ValueError(f"The type of operator '{type_of_operator}' is not recognized.")
    elif dim==3:
        if type_of_operator=='D':
            return nn.Sequential(
                deconv(64, 32, **cfg_decoder),
                nn.ReLU(),
                nn.Upsample(scale_factor=4, **cfg_upscale), 
                deconv(32, 16, **cfg_decoder), 
                nn.ReLU(),
                nn.Upsample(scale_factor=4, **cfg_upscale), 
                deconv(16, 9, **cfg_decoder),
            )
        elif type_of_operator=='F':
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
        else:
            raise ValueError(f"The type of operator '{type_of_operator}' is not recognized.")
    else:
        raise NotImplementedError(f"Dimension {dim} is not supported.")
    
    
##################################################
##################################################
class CNN_enc_dec(nn.Module):
    """The convolutional encoder-decoder architecture for approximating the collision operators defined *for a fixed resolution*.
    The architecture is same as the one used in [opPINN: Physics-informed neural network with operator learning to approximate solutions to the Fokker-Planck-Landau equation](https://www.sciencedirect.com/science/article/pii/S0021999123001262).
    """
    def __init__(
            self,
            dim:                int,
            res:                int,
            type_of_operator:   str,
            encoder:    Optional[nn.Sequential] = None,
            decoder:    Optional[nn.Sequential] = None,
            device:     Optional[torch.device] = None,
        ) -> Self:
        assert dim in (2, 3), "Only 2D and 3D are supported."
        assert type_of_operator in ('D', 'F'), "'type_of_operator' should be 'D' or 'F'."
        super().__init__()
        if device is None:  device = torch.get_default_device()
        
        self.__dim:             int = dim
        self.__res:             int = res
        
        self.encoder: Optional[nn.Sequential] = encoder
        self.decoder: Optional[nn.Sequential] = decoder
        if encoder is None: self.encoder = _get_base_encoder(dim, res)
        if decoder is None: self.decoder = _get_base_decoder(dim, type_of_operator)
        self.to(device)
        return
    
    
    def forward(self, f: torch.Tensor) -> torch.Tensor:
        f = self.encoder.forward(f)
        f = f.view(f.shape[0], 64, *(self.__res//2**4 for _ in range(self.__dim)))
        return self.decoder.forward(f)


def generate_operator_D(dim: int, res: int, device: Optional[torch.device]=None) -> CNN_enc_dec:
    return CNN_enc_dec(dim, res, type_of_operator='D', device=device)
def generate_operator_F(dim: int, res: int, device: Optional[torch.device]=None) -> CNN_enc_dec:
    return CNN_enc_dec(dim, res, type_of_operator='F', device=device)


class NeuralCollisionOperator():
    def __init__(
            self,
            dimension:  int,
            resolution: int,
            v_max:      float,
            op_D:       torch.nn.Module,
            op_F:       torch.nn.Module,
            coeff:      float   = 1.0,
            device:     Optional[torch.device]  = None,
        ) -> Self:
        from    utils   import  FiniteDifferenceMethod
        device = device if device is not None else torch.get_default_device()
        self.__dim = dimension
        self.__res = resolution
        self.op_D = op_D.to(device)
        self.op_F = op_F.to(device)
        self.op_D.eval()
        self.op_F.eval()
        self.__coeff = coeff
        self.__shape_tv = tuple([-1, *(resolution for _ in range(dimension))])
        self.__fdm = FiniteDifferenceMethod(dimension, 2*v_max/resolution, device=device)
        return
    
    
    def forward(self, f: torch.Tensor, grad_f: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            `f` (`torch.Tensor`):
                A tensor of shape `(num_points, 1)` aligned in the `ij`-indexing style.
                Here, `num_points` should be equal to `v_num_grid**dimension`.
            `grad_f` (`torch.Tensor`):
                A tensor of shape `(num_points, dimension)` aligned in the `ij`-indexing style.
                This tensor saves the spatial gradient of the temporal distribution `f` *without the temporal derivative*.
                Here, `num_points` should be equal to `v_num_grid**dimension`.
        
        Remark:
            Do not forget to detach the input tensors before passing them to this method. Otherwise, the computational graph will be accumulated.
        """
        old_shape = f.shape
        dim, res = self.__dim, self.__res
        f = f.reshape(self.__shape_tv)
        grad_f = grad_f.reshape(*(f.shape), dim)
        
        Df: torch.Tensor = self.op_D.forward(f[:, None])  # Unsqueeze at `dim=1`
        Ff: torch.Tensor = self.op_F.forward(f[:, None])  # Unsqueeze at `dim=1`
        Df = Df.reshape(-1, dim, dim, *(res for _ in range(dim)))
        Ff = Ff.reshape(-1, dim, *(res for _ in range(dim)))
        
        # So far, the tensors to be used are reshaped as follows:
        # * f:        (num_t, *domain)
        # * grad_f:   (num_t, *domain, dim)
        # * Df:       (num_t, dim, dim, *domain)
        # * Ff:       (num_t, dim, *domain)
        operands: list[torch.Tensor] = [
            torch.einsum("tj..., t...j -> t...", Df[:, d], grad_f) - Ff[:, d]*f
            for d in range(dim)
        ]
        # `operands` saves tensors of shape `(num_t, *domain)`,
        # and the `i`-th tensor will be differentiated with respect to `v_i`
        operands = [
            self.__fdm.compute_derivative(_op, _idx)
            for _idx, _op in enumerate(operands)
        ]
        q = torch.stack(operands, dim=-1).sum(dim=-1)
        return self.__coeff * q.reshape(old_shape)


##################################################
##################################################
# End of file