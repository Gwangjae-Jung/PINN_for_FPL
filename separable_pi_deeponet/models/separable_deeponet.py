from   typing      import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import torch
import torch.nn as nn

from   .base       import BaseModule
from   ..autograd  import derivatives


__all__: List[str] = ['SeparableDeepONet', 'trunk_r_einsum']


_RESERVED_INDICES: set[str] = {'B', 'P', 'R', 'C'}
_AVAILABLE_INDICES: List[str] = [chr(c) for c in range(ord('A'), ord('Z') + 1) if chr(c) not in _RESERVED_INDICES]


##################################################
def trunk_r_einsum(trunk_outputs: Sequence[torch.Tensor]) -> torch.Tensor:
    """Combines a sequence of trunk network outputs by contracting over tensor rank `r`.

    ## Description
    Given $d$ trunk outputs, each of shape `(N_j, r, p, c)`, this function computes their outer product
    contracted over the tensor rank dimension $r$ via Einstein summation.
    The resulting tensor has shape `(N_1, N_2, ..., N_d, p, c)`.

    Arguments:
        `trunk_outputs` (`Sequence[torch.Tensor]`): Sequence of $d$ tensors, each of shape `(N_j, r, p, c)`.

    Returns:
        `torch.Tensor`: Combined trunk output tensor of shape `(N_1, N_2, ..., N_d, p, c)`.
    """
    n_trunks: int = len(trunk_outputs)
    if n_trunks == 0:
        raise ValueError("trunk_outputs sequence must contain at least one tensor.")

    trunk_chars: List[str] = [_AVAILABLE_INDICES[i] for i in range(n_trunks)]
    lhs_terms:   List[str] = [f"{char}RPC" for char in trunk_chars]
    rhs_term:    str       = f"{''.join(trunk_chars)}PC"
    einsum_cmd:  str       = f"{','.join(lhs_terms)}->{rhs_term}"

    return torch.einsum(einsum_cmd, *trunk_outputs)


class SeparableDeepONet(BaseModule):
    """Separable Physics-Informed Deep Operator Network (Sep-PI-DeepONet) in PyTorch.

    ## Description
    Separable DeepONet utilizes factorized coordinate representations and separate 1D sub-networks
    for each coordinate axis to break the curse of dimensionality in operator learning.
    Forward-mode automatic differentiation is integrated to evaluate derivatives of the 1D trunk networks,
    enabling linear scaling with grid resolution and dimension.

    -----
    ### References
    [1] Luis Mandl, Somdatta Goswami, Lena Lambers, and Tim Ricken.
        Separable physics-informed DeepONet: Breaking the curse of dimensionality in physics-informed machine learning.
        Computer Methods in Applied Mechanics and Engineering, 434:117586, 2025.
        https://doi.org/10.1016/j.cma.2024.117586
    [2] Junwoo Cho, Seungtae Nam, Hyunmo Yang, Seok-Bae Yun, Youngjoon Hong, and Eunbyung Park.
        Separable physics-informed neural networks.
        Advances in Neural Information Processing Systems (NeurIPS), 2023.
        https://neurips.cc/virtual/2022/59890
    """

    def __init__(
            self,
            branch_layers: Sequence[int],
            trunk_layers:  Sequence[int],
            n_trunk_vars:  int            = 2,
            r:             int            = 128,
            split_branch:  bool           = False,
            split_trunk:   bool           = False,
            stacked:       bool           = False,
            output_dim:    int            = 1,
            deg_grad:      Sequence[int]  = [],
            activation:    str            = 'tanh',
            bias:          bool           = True,
        ) -> None:
        """Initializes the `SeparableDeepONet` module.

        Arguments:
            `branch_layers` (`Sequence[int]`): Layer width sequence for the branch network.
            `trunk_layers` (`Sequence[int]`): Layer width sequence for each 1D trunk subnetwork.
            `n_trunk_vars` (`int`, default: `2`): Number of independent 1D trunk input coordinate variables ($d$).
            `r` (`int`, default: `128`): Tensor decomposition rank for outer product representation.
            `split_branch` (`bool`, default: `False`): Flag determining whether branch features are split for multi-output fields.
            `split_trunk` (`bool`, default: `False`): Flag determining whether trunk features are split for multi-output fields.
            `stacked` (`bool`, default: `False`): Flag indicating whether stacked branch subnetworks are employed.
            `output_dim` (`int`, default: `1`): Number of output solution fields or channels ($c$).
            `deg_grad` (`Sequence[int]`, default: `[]`): Differentiation degree for each coordinate variable for forward-mode autograd.
            `activation` (`str`, default: `'tanh'`): Nonlinear activation function name (`'tanh'`, `'relu'`, or `'sigmoid'`).
            `bias` (`bool`, default: `True`): Flag determining whether a learnable output bias vector is included.

        Returns:
            This method returns no values.
        """
        super().__init__()
        self.branch_layers: List[int]       = list(branch_layers)
        self.trunk_layers:  List[int]       = list(trunk_layers)
        self.n_trunk_vars:  int             = n_trunk_vars
        self.r:             int             = r
        self.split_branch:  bool            = split_branch
        self.split_trunk:   bool            = split_trunk
        self.stacked:       bool            = stacked
        self.output_dim:    int             = output_dim
        self.act_name:      str             = activation
        self.deg_grad:      Tuple[int, ...] = tuple(deg_grad) if deg_grad else tuple([0] * n_trunk_vars)

        self.act_fn: nn.Module = self._get_activation(activation)

        # Build branch network
        if self.stacked:
            self.branch_nets: nn.ModuleList = nn.ModuleList()
            num_stacked: int = self.branch_layers[-1]
            for _ in range(num_stacked):
                sub_layers: List[nn.Module] = []
                for in_f, out_f in zip(self.branch_layers[:-2], self.branch_layers[1:-1]):
                    linear = nn.Linear(in_f, out_f)
                    self._init_weights(linear)
                    sub_layers.append(linear)
                    sub_layers.append(self._get_activation(activation))
                last_linear = nn.Linear(self.branch_layers[-2], 1)
                self._init_weights(last_linear)
                sub_layers.append(last_linear)
                self.branch_nets.append(nn.Sequential(*sub_layers))
            self.branch_net = None
        else:
            branch_sub_layers: List[nn.Module] = []
            for in_f, out_f in zip(self.branch_layers[:-2], self.branch_layers[1:-1]):
                linear = nn.Linear(in_f, out_f)
                self._init_weights(linear)
                branch_sub_layers.append(linear)
                branch_sub_layers.append(self._get_activation(activation))
            last_linear = nn.Linear(self.branch_layers[-2], self.branch_layers[-1])
            self._init_weights(last_linear)
            branch_sub_layers.append(last_linear)
            self.branch_net = nn.Sequential(*branch_sub_layers)
            self.branch_nets = None

        # Build d independent 1D trunk sub-networks
        # Each trunk maps 1D coordinate to r * hidden_dim (* output_dim if split_trunk)
        self.trunk_nets: nn.ModuleList = nn.ModuleList()
        for j in range(self.n_trunk_vars):
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

        # Build derivative operators for forward-mode autograd
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
        """Evaluates forward pass of the branch network.

        Arguments:
            `branch_x` (`torch.Tensor`): Input functions evaluated at sensor points, shape `(batch_size_b, sensors)`.

        Returns:
            `torch.Tensor`: Branch embedding tensor of shape `(batch_size_b, p, output_dim)`.
        """
        if self.stacked:
            branch_outs: List[torch.Tensor] = [net(branch_x) for net in self.branch_nets]
            b_out: torch.Tensor = torch.cat(branch_outs, dim=-1)
        else:
            b_out = self.branch_net(branch_x)

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
        """Evaluates the Separable DeepONet forward pass.

        ## Description
        Computes branch embeddings and separable 1D trunk embeddings, caches intermediate derivatives
        if differentiation is enabled, and contracts across rank $r$ and latent dimension $p$ via einsum.

        Arguments:
            `branch_x` (`torch.Tensor`): Branch input functions, shape `(batch_size_b, sensors)`.
            `*trunk_x` (`torch.Tensor`): Sequence of $d$ 1D coordinate tensors, each of shape `(N_j, 1)`.
            `compute_derivatives` (`Optional[bool]`, default: `None`): Flag to compute and cache trunk derivatives.
                If `None`, defaults to `self.training and any(d > 0 for d in self.deg_grad)`.

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

        ## Description
        Evaluates partial derivatives of any order with respect to the trunk coordinates without reverse-mode AD,
        by selecting the appropriate precomputed trunk derivative orders and executing the contracted Einstein summation.

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
