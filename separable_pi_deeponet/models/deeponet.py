from   typing      import List, Optional, Sequence, Union
import torch
import torch.nn as nn

from   .base       import BaseModule


__all__: List[str] = ['DeepONet']


##################################################
class DeepONet(BaseModule):
    """Conventional (Vanilla) Deep Operator Network (DeepONet) in PyTorch.

    ## Description
    DeepONet approximates nonlinear operators mapping from infinite-dimensional function spaces
    to function spaces. The architecture comprises a branch network encoding input functions evaluated at sensor points
    and a trunk network encoding spatiotemporal coordinates.

    -----
    ### References
    [1] Lu Lu, Pengzhan Jin, Guofei Pang, Zhongqiang Zhang, and George Em Karniadakis.
        Learning nonlinear operators via DeepONet based on the universal approximation theorem of operators.
        Nature Machine Intelligence, 3(3):218–229, 2021.
    [2] Sifan Wang, Hanwen Wang, and Paris Perdikaris.
        Learning the solution operator of parametric partial differential equations with physics-informed DeepONets.
        Science Advances, 7(40):eabi8605, 2021.
    """

    def __init__(
            self,
            branch_layers: Sequence[int],
            trunk_layers:  Sequence[int],
            split_branch:  bool           = False,
            split_trunk:   bool           = False,
            stacked:       bool           = False,
            output_dim:    int            = 1,
            activation:    str            = 'tanh',
            bias:          bool           = True,
        ) -> None:
        """Initializes the `DeepONet` module.

        Arguments:
            `branch_layers` (`Sequence[int]`): Layer width sequence for the branch network, beginning with sensor input features.
            `trunk_layers` (`Sequence[int]`): Layer width sequence for the trunk network, beginning with coordinate dimension.
            `split_branch` (`bool`, default: `False`): Flag determining whether branch output features are split among output dimensions.
            `split_trunk` (`bool`, default: `False`): Flag determining whether trunk output features are split among output dimensions.
            `stacked` (`bool`, default: `False`): Flag indicating whether stacked independent branch subnetworks are employed.
            `output_dim` (`int`, default: `1`): Number of output solution fields or channels.
            `activation` (`str`, default: `'tanh'`): Nonlinear activation function name (`'tanh'`, `'relu'`, or `'sigmoid'`).
            `bias` (`bool`, default: `True`): Flag determining whether a learnable output bias vector is included.

        Returns:
            This method returns no values.
        """
        super().__init__()
        self.branch_layers: List[int] = list(branch_layers)
        self.trunk_layers:  List[int] = list(trunk_layers)
        self.split_branch:  bool      = split_branch
        self.split_trunk:   bool      = split_trunk
        self.stacked:       bool      = stacked
        self.output_dim:    int       = output_dim
        self.act_name:      str       = activation

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

        # Build trunk network
        trunk_sub_layers: List[nn.Module] = []
        for in_f, out_f in zip(self.trunk_layers[:-2], self.trunk_layers[1:-1]):
            linear = nn.Linear(in_f, out_f)
            self._init_weights(linear)
            trunk_sub_layers.append(linear)
            trunk_sub_layers.append(self._get_activation(activation))
        last_trunk_linear = nn.Linear(self.trunk_layers[-2], self.trunk_layers[-1])
        self._init_weights(last_trunk_linear)
        trunk_sub_layers.append(last_trunk_linear)
        trunk_sub_layers.append(self._get_activation(activation))
        self.trunk_net: nn.Sequential = nn.Sequential(*trunk_sub_layers)

        # Output bias parameter
        if bias:
            self.output_bias: Optional[nn.Parameter] = nn.Parameter(torch.zeros(self.output_dim))
        else:
            self.register_parameter('output_bias', None)
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
        """Initializes linear layer weights with Glorot normal (Xavier normal) distribution.

        Arguments:
            `layer` (`nn.Linear`): Linear layer to initialize.

        Returns:
            This method returns no values.
        """
        nn.init.xavier_normal_(layer.weight)
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)
        return None

    def forward(
            self,
            branch_x: torch.Tensor,
            trunk_x:  torch.Tensor,
        ) -> torch.Tensor:
        """Evaluates the DeepONet forward pass.

        ## Description
        Passes `branch_x` through the branch network and `trunk_x` through the trunk network,
        reshapes representations according to `split_branch` and `split_trunk`, and merges representations via einsum.

        Arguments:
            `branch_x` (`torch.Tensor`): Input functions evaluated at sensors, shape `(batch_size_b, branch_in_dim)`.
            `trunk_x` (`torch.Tensor`): Spatiotemporal coordinate points, shape `(batch_size_t, trunk_in_dim)`.

        Returns:
            `torch.Tensor`: Predicted operator solution tensor.
        """
        # 1. Branch forward
        if self.stacked:
            branch_outs: List[torch.Tensor] = [net(branch_x) for net in self.branch_nets]
            b_out: torch.Tensor = torch.cat(branch_outs, dim=-1)
        else:
            b_out = self.branch_net(branch_x)

        if self.split_branch:
            b_out = b_out.reshape(b_out.shape[0], -1, self.output_dim)
        else:
            b_out = b_out.reshape(b_out.shape[0], -1, 1)

        # 2. Trunk forward
        t_out: torch.Tensor = self.trunk_net(trunk_x)

        if self.split_trunk:
            t_out = t_out.reshape(t_out.shape[0], -1, self.output_dim)
        else:
            t_out = t_out.reshape(t_out.shape[0], -1, 1)

        # 3. Einsum combination
        if b_out.shape[0] == t_out.shape[0] and b_out.shape[0] > 1:
            # Pointwise paired samples: [batch_size, p, output_dim] * [batch_size, p, output_dim] -> [batch_size, output_dim]
            result: torch.Tensor = torch.einsum('ijk,ijk->ik', b_out, t_out)
        else:
            # All combinations: [batch_branch, p, output_dim] * [batch_trunk, p, output_dim] -> [batch_branch, batch_trunk, output_dim]
            result = torch.einsum('ipk,jpk->ijk', b_out, t_out)

        if self.output_bias is not None:
            result = result + self.output_bias

        return result


##################################################
def main() -> None:
    return None


if __name__ == '__main__':
    main()
