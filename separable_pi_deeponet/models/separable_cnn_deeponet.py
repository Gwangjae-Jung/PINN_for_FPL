from   typing      import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import torch
import torch.nn as nn

from   .base       import BaseModule
from   ..autograd  import derivatives


__all__: List[str] = ['SeparableBranchCNNDeepONet']


_RESERVED_INDICES: set[str] = {'B', 'P', 'R', 'C'}
_AVAILABLE_INDICES: List[str] = [chr(c) for c in range(ord('A'), ord('Z') + 1) if chr(c) not in _RESERVED_INDICES]


##################################################
class SeparableBranchCNNDeepONet(BaseModule):
    """Separable Physics-Informed Deep Operator Network with CNN branch network.

    ## Description
    Employs a 2D Convolutional Neural Network (CNN) as the branch network to encode spatial field inputs
    (e.g., random Gaussian fields, source distributions) and a separable 1D trunk network for coordinates.
    Computes operator solutions with outer product and summation over tensor rank $r$ and latent dimension $p$.

    -----
    ### References
    [1] Luis Mandl, Somdatta Goswami, Lena Lambers, and Tim Ricken.
        Separable physics-informed DeepONet: Breaking the curse of dimensionality in physics-informed machine learning.
        Computer Methods in Applied Mechanics and Engineering, 434:117586, 2025.
        https://doi.org/10.1016/j.cma.2024.117586
    """

    def __init__(
            self,
            cnn_branch_blocks: Sequence[Sequence[Any]],
            trunk_layers:      Sequence[int],
            in_channels:       int           = 1,
            n_trunk_vars:      int           = 3,
            r:                 int           = 80,
            split_branch:      bool          = False,
            split_trunk:       bool          = False,
            stacked:           bool          = False,
            output_dim:        int           = 1,
            deg_grad:          Sequence[int] = [],
            activation:        str           = 'tanh',
            bias:              bool          = True,
        ) -> None:
        """Initializes the `SeparableBranchCNNDeepONet` module.

        Arguments:
            `cnn_branch_blocks` (`Sequence[Sequence[Any]]`): Structural specifications of the CNN branch blocks.
            `trunk_layers` (`Sequence[int]`): Layer width sequence for each 1D trunk subnetwork.
            `in_channels` (`int`, default: `1`): Number of channels in the 2D input field.
            `n_trunk_vars` (`int`, default: `3`): Number of 1D trunk input coordinate variables ($d$).
            `r` (`int`, default: `80`): Tensor decomposition rank for separable outer products.
            `split_branch` (`bool`, default: `False`): Flag determining whether branch features are split across output dimensions.
            `split_trunk` (`bool`, default: `False`): Flag determining whether trunk features are split across output dimensions.
            `stacked` (`bool`, default: `False`): Flag indicating whether stacked branch subnetworks are employed.
            `output_dim` (`int`, default: `1`): Number of output solution fields or channels ($c$).
            `deg_grad` (`Sequence[int]`, default: `[]`): Differentiation degree for each coordinate variable.
            `activation` (`str`, default: `'tanh'`): Nonlinear activation function name (`'tanh'`, `'relu'`, or `'sigmoid'`).
            `bias` (`bool`, default: `True`): Flag determining whether a learnable output bias vector is included.

        Returns:
            This method returns no values.
        """
        super().__init__()
        self.cnn_branch_blocks: List[List[Any]] = [list(block) for block in cnn_branch_blocks]
        self.trunk_layers:      List[int]       = list(trunk_layers)
        self.in_channels:       int             = in_channels
        self.n_trunk_vars:      int             = n_trunk_vars
        self.r:                 int             = r
        self.split_branch:      bool            = split_branch
        self.split_trunk:       bool            = split_trunk
        self.stacked:           bool            = stacked
        self.output_dim:        int             = output_dim
        self.act_name:          str             = activation
        self.deg_grad:          Tuple[int, ...] = tuple(deg_grad) if deg_grad else tuple([0] * n_trunk_vars)

        self.act_fn: nn.Module = self._get_activation(activation)

        # Build CNN branch network
        self.branch_net: nn.Module = self._build_cnn_branch(self.cnn_branch_blocks, in_channels)

        # Build d independent 1D trunk sub-networks
        self.trunk_nets: nn.ModuleList = nn.ModuleList()
        for _ in range(self.n_trunk_vars):
            trunk_sub_layers: List[nn.Module] = []
            in_dim: int = self.trunk_layers[0] if len(self.trunk_layers) > 0 else 1
            for in_f, out_f in zip([in_dim] + self.trunk_layers[:-2], self.trunk_layers[:-1]):
                linear = nn.Linear(in_f, out_f)
                self._init_weights(linear)
                trunk_sub_layers.append(linear)
                trunk_sub_layers.append(self._get_activation(activation))
            out_features: int = self.trunk_layers[-1] * self.r
            last_trunk_linear = nn.Linear(self.trunk_layers[-2], out_features)
            self._init_weights(last_trunk_linear)
            trunk_sub_layers.append(last_trunk_linear)
            trunk_sub_layers.append(self._get_activation(activation))
            self.trunk_nets.append(nn.Sequential(*trunk_sub_layers))

        # Output bias
        if bias:
            self.output_bias: Optional[nn.Parameter] = nn.Parameter(torch.zeros(self.output_dim))
        else:
            self.register_parameter('output_bias', None)

        # Derivative operators for forward-mode autograd
        self.__diff_ops: List[Callable[[torch.Tensor], List[torch.Tensor]]] = [
            derivatives(net, deg) for net, deg in zip(self.trunk_nets, self.deg_grad)
        ]
        self.__trunk_outputs: List[List[torch.Tensor]] = []
        self.__branch_output: Optional[torch.Tensor]   = None
        self.__einsum_cmd:    str                      = ""
        self._setup_einsum_commands()
        return None

    @staticmethod
    def _get_activation(name: str) -> nn.Module:
        """Retrieves activation module corresponding to `name`.

        Arguments:
            `name` (`str`): Name of the activation function.

        Returns:
            `nn.Module`: PyTorch activation module.
        """
        lower: str = name.lower()
        if lower == 'tanh':
            return nn.Tanh()
        elif lower == 'relu':
            return nn.ReLU()
        elif lower == 'sigmoid':
            return nn.Sigmoid()
        elif lower == 'gelu':
            return nn.GELU()
        elif lower in ('none', 'identity'):
            return nn.Identity()
        else:
            raise ValueError(f"Unsupported activation function: '{name}'.")

    @staticmethod
    def _init_weights(layer: nn.Linear) -> None:
        """Initializes linear layer weights with Glorot normal distribution.

        Arguments:
            `layer` (`nn.Linear`): Linear layer to initialize.

        Returns:
            This method returns no values.
        """
        nn.init.xavier_normal_(layer.weight)
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)
        return None

    def _build_cnn_branch(self, blocks: List[List[Any]], in_channels: int) -> nn.Sequential:
        """Builds a sequential CNN branch network from architectural block specifications.

        Arguments:
            `blocks` (`List[List[Any]]`): Sequence of block specifications.
            `in_channels` (`int`): Input channel count for the first convolutional layer.

        Returns:
            `nn.Sequential`: Constructed PyTorch CNN sequential module.
        """
        layers:       List[nn.Module] = []
        current_in_c: int             = in_channels

        for block in blocks:
            b_len: int = len(block)
            if b_len == 4:
                # Conv block: [out_channels, kernel_h, kernel_w, activation]
                out_c, kh, kw, act = block
                conv = nn.Conv2d(current_in_c, out_c, kernel_size=(kh, kw), padding='same')
                nn.init.xavier_normal_(conv.weight)
                if conv.bias is not None:
                    nn.init.zeros_(conv.bias)
                layers.append(conv)
                layers.append(self._get_activation(str(act)))
                current_in_c = out_c
            elif b_len == 5:
                # Pool block: [pool_type, window_h, window_w, stride_h, stride_w]
                ptype, wh, ww, sh, sw = block
                if ptype == 'avg_pool':
                    layers.append(nn.AvgPool2d(kernel_size=(wh, ww), stride=(sh, sw)))
                elif ptype == 'max_pool':
                    layers.append(nn.MaxPool2d(kernel_size=(wh, ww), stride=(sh, sw)))
                else:
                    raise ValueError(f"Unknown pooling type: '{ptype}'.")
            elif b_len == 1 and block[0] == 'flatten':
                # Flatten block
                layers.append(nn.Flatten())
            elif b_len == 2:
                # Dense block: [out_features, activation]
                out_f, act = block
                # We use LazyLinear or dynamic linear on first forward if input dimension is not predetermined
                linear = nn.LazyLinear(out_f)
                layers.append(linear)
                if str(act).lower() != 'none':
                    layers.append(self._get_activation(str(act)))
            else:
                raise ValueError(f"Invalid block specification: {block}")

        return nn.Sequential(*layers)

    def _setup_einsum_commands(self) -> None:
        """Constructs Einstein summation command strings for forward passes and gradient evaluation.

        Returns:
            This method returns no values.
        """
        trunk_chars: List[str] = [_AVAILABLE_INDICES[i] for i in range(self.n_trunk_vars)]
        lhs_terms:   List[str] = ['BPC'] + [f"{char}RPC" for char in trunk_chars]
        rhs_term:    str       = f"B{''.join(trunk_chars)}C"
        self.__einsum_cmd = f"{','.join(lhs_terms)}->{rhs_term}"
        return None

    @property
    def trunk_outputs(self) -> List[List[torch.Tensor]]:
        """List of cached trunk network outputs and evaluated derivatives."""
        return self.__trunk_outputs

    @property
    def branch_output(self) -> Optional[torch.Tensor]:
        """Cached branch network output tensor."""
        return self.__branch_output

    def forward_branch(self, branch_x: torch.Tensor) -> torch.Tensor:
        """Evaluates forward pass of the CNN branch network.

        Arguments:
            `branch_x` (`torch.Tensor`): Input 2D field tensor.

        Returns:
            `torch.Tensor`: Branch embedding tensor of shape `(batch_size_b, p, output_dim)`.
        """
        # Ensure shape (batch, channels, H, W)
        x: torch.Tensor = branch_x
        if x.ndim == 3:
            # (batch, H, W) -> (batch, 1, H, W)
            x = x.unsqueeze(1)
        elif x.ndim == 4 and x.shape[-1] == self.in_channels and x.shape[1] != self.in_channels:
            # (batch, H, W, channels) -> (batch, channels, H, W)
            x = x.permute(0, 3, 1, 2)

        b_out: torch.Tensor = self.branch_net(x)

        if self.split_branch:
            b_out = b_out.reshape(b_out.shape[0], -1, self.output_dim)
        else:
            b_out = b_out.reshape(b_out.shape[0], -1, 1)

        return b_out

    def forward(
            self,
            branch_x:            torch.Tensor,
            *trunk_x:            torch.Tensor,
            compute_derivatives: Optional[bool] = None,
        ) -> torch.Tensor:
        """Evaluates the Separable CNN DeepONet forward pass.

        Arguments:
            `branch_x` (`torch.Tensor`): Branch input 2D field tensor.
            `*trunk_x` (`torch.Tensor`): Sequence of $d$ 1D coordinate tensors, each of shape `(N_j, 1)`.
            `compute_derivatives` (`Optional[bool]`, default: `None`): Flag to compute and cache trunk derivatives.

        Returns:
            `torch.Tensor`: Evaluated solution tensor on the tensor lattice grid.
        """
        if len(trunk_x) != self.n_trunk_vars:
            raise ValueError(f"Expected {self.n_trunk_vars} coordinate tensors, but received {len(trunk_x)}.")

        if compute_derivatives is None:
            compute_derivatives = self.training and any(d > 0 for d in self.deg_grad)

        # 1. Forward branch
        b_out: torch.Tensor = self.forward_branch(branch_x)
        self.__branch_output = b_out

        # 2. Forward trunks
        self.__trunk_outputs.clear()
        trunk_operands: List[torch.Tensor] = []

        for idx, (net, x_i) in enumerate(zip(self.trunk_nets, trunk_x)):
            x_in: torch.Tensor = x_i.reshape(-1, 1) if x_i.ndim == 1 else x_i
            res:  int          = x_in.shape[0]

            if compute_derivatives and self.deg_grad[idx] > 0:
                ops = self.__diff_ops[idx]
                deriv_list: List[torch.Tensor] = ops(x_in)
                formatted_derivs: List[torch.Tensor] = []
                for d_out in deriv_list:
                    if self.split_trunk:
                        formatted = d_out.reshape(res, self.r, -1, self.output_dim)
                    else:
                        formatted = d_out.reshape(res, self.r, -1, 1)
                    formatted_derivs.append(formatted)
                self.__trunk_outputs.append(formatted_derivs)
                trunk_operands.append(formatted_derivs[0])
            else:
                out_i: torch.Tensor = net(x_in)
                if self.split_trunk:
                    formatted_out = out_i.reshape(res, self.r, -1, self.output_dim)
                else:
                    formatted_out = out_i.reshape(res, self.r, -1, 1)
                self.__trunk_outputs.append([formatted_out])
                trunk_operands.append(formatted_out)

        # 3. Contract over r and p
        result: torch.Tensor = torch.einsum(self.__einsum_cmd, b_out, *trunk_operands)

        if self.output_bias is not None:
            result = result + self.output_bias

        return result

    def compute_gradient(self, deg_grad: Sequence[int]) -> torch.Tensor:
        """Computes mixed partial derivatives of the separable operator using cached trunk derivatives.

        Arguments:
            `deg_grad` (`Sequence[int]`): Differentiation order for each coordinate axis ($d_1, d_2, \\dots, d_d$).

        Returns:
            `torch.Tensor`: The evaluated partial derivative tensor on the coordinate lattice.
        """
        if len(deg_grad) != self.n_trunk_vars:
            raise ValueError(f"Length of deg_grad ({len(deg_grad)}) must match n_trunk_vars ({self.n_trunk_vars}).")
        if self.__branch_output is None:
            raise RuntimeError("Must execute forward() with compute_derivatives=True before calling compute_gradient().")

        operands: List[torch.Tensor] = []
        for idx, d in enumerate(deg_grad):
            if d >= len(self.__trunk_outputs[idx]):
                raise ValueError(
                    f"Variable {idx} requested derivative degree {d}, but only up to degree "
                    f"{len(self.__trunk_outputs[idx]) - 1} was computed during forward pass."
                )
            operands.append(self.__trunk_outputs[idx][d])

        result: torch.Tensor = torch.einsum(self.__einsum_cmd, self.__branch_output, *operands)
        return result


##################################################
def main() -> None:
    return None


if __name__ == '__main__':
    main()
