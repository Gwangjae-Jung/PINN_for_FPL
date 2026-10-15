from   typing                  import Any, Dict, List, Optional, Sequence, Tuple, Union
from   typing_extensions       import TypedDict
import torch
import torch.nn as nn

from   .deeponet               import DeepONet
from   .separable_deeponet     import SeparableDeepONet
from   .separable_cnn_deeponet import SeparableBranchCNNDeepONet


__all__: List[str] = ['DeepONetConfig', 'setup_deeponet']


##################################################
class DeepONetConfig(TypedDict, total=False):
    """Configuration dictionary for initializing DeepONet models.

    ## Description
    Specifies architectural hyperparameters for DeepONet and Separable DeepONet variants.
    """
    num_outputs:                int
    """Number of output target fields or channels."""
    split_branch:               bool
    """Whether to split branch features across output dimensions."""
    split_trunk:                bool
    """Whether to split trunk features across output dimensions."""
    separable:                  bool
    """Whether to instantiate a separable DeepONet architecture."""
    r:                          int
    """Tensor rank approximation parameter for separable trunk."""
    hidden_dim:                 int
    """Latent embedding dimension $p$."""
    n_sensors:                  int
    """Number of sensor measurement points for branch input function."""
    branch_input_features:      int
    """Feature dimensionality per branch sensor."""
    trunk_input_features:       int
    """Dimensionality or count of trunk coordinate variables."""
    branch_layers:              Sequence[int]
    """Hidden layer width sequence for MLP branch."""
    trunk_layers:               Sequence[int]
    """Hidden layer width sequence for trunk network(s)."""
    stacked_do:                 bool
    """Whether to construct stacked independent branch networks."""
    branch_cnn:                 bool
    """Whether to use a Convolutional Neural Network branch."""
    branch_cnn_blocks:          Sequence[Sequence[Any]]
    """Block specifications for CNN branch."""
    branch_cnn_input_size:      Sequence[int]
    """Spatial resolution (H, W) of input 2D field."""
    branch_cnn_input_channels:  int
    """Channel count of 2D input field."""
    deg_grad:                   Sequence[int]
    """Maximum differentiation order for each coordinate axis."""
    activation:                 str
    """Nonlinear activation function name ('tanh', 'relu', etc.)."""
    bias:                       bool
    """Whether to include a learnable output bias vector."""
    total_params:               int
    """Cached total number of trainable parameters."""


def setup_deeponet(
    config: Union[DeepONetConfig, Dict[str, Any], Any],
) -> Tuple[Union[DeepONetConfig, Any], nn.Module]:
    """Configures, constructs, and initializes a DeepONet or Separable DeepONet model.

    ## Description
    Validates architectural specifications, adjusts layer dimensions for output splitting,
    instantiates the appropriate PyTorch module (`DeepONet`, `SeparableDeepONet`, or `SeparableBranchCNNDeepONet`),
    calculates trainable parameters, and displays an architectural summary.

    Arguments:
        `config` (`Union[DeepONetConfig, Dict[str, Any], Any]`): Configuration dictionary or `argparse.Namespace`.

    Returns:
        `Tuple[Union[DeepONetConfig, Any], nn.Module]`: Tuple containing the updated configuration and initialized model.
    """
    # Helper to get configuration values from dict or object
    def get_val(key: str, default: Any = None) -> Any:
        if isinstance(config, dict):
            return config.get(key, default)
        return getattr(config, key, default)

    def set_val(key: str, value: Any) -> None:
        if isinstance(config, dict):
            config[key] = value
        else:
            setattr(config, key, value)
        return None

    num_outputs:           int           = get_val('num_outputs', 1)
    split_branch:          bool          = get_val('split_branch', False)
    split_trunk:           bool          = get_val('split_trunk', False)
    separable:             bool          = get_val('separable', True)
    r:                     int           = get_val('r', 128)
    hidden_dim:            int           = get_val('hidden_dim', 100)
    n_sensors:             int           = get_val('n_sensors', 100)
    branch_in_feat:        int           = get_val('branch_input_features', 1)
    trunk_in_feat:         int           = get_val('trunk_input_features', 2)
    stacked_do:            bool          = get_val('stacked_do', False)
    branch_cnn:            bool          = get_val('branch_cnn', False)
    activation:            str           = get_val('activation', 'tanh')
    bias:                  bool          = get_val('bias', True)
    deg_grad:              Sequence[int] = get_val('deg_grad', [])

    # Validation
    if num_outputs == 1:
        split_trunk  = False
        split_branch = False
        set_val('split_trunk', False)
        set_val('split_branch', False)

    if num_outputs > 1 and not split_trunk and not split_branch:
        raise ValueError("split_trunk and split_branch cannot both be False for multi-output models.")

    if separable and r <= 0:
        raise ValueError(f"'r' must be a positive integer for separable DeepONet, received {r}.")

    raw_trunk_layers = get_val('trunk_layers', [100, 100, 100, 100, 100])
    raw_branch_layers = get_val('branch_layers', [100, 100, 100, 100, 100])

    trunk_layers_list: List[int] = [raw_trunk_layers] if isinstance(raw_trunk_layers, int) else list(raw_trunk_layers)
    branch_layers_list: List[int] = [raw_branch_layers] if isinstance(raw_branch_layers, int) else list(raw_branch_layers)

    # Add output features to trunk layers
    if split_trunk:
        trunk_layers = trunk_layers_list + [num_outputs * hidden_dim]
    else:
        trunk_layers = trunk_layers_list + [hidden_dim]

    model: nn.Module
    if not separable:
        # Vanilla DeepONet
        if split_branch:
            branch_layers = [n_sensors * branch_in_feat] + branch_layers_list + [num_outputs * hidden_dim]
        else:
            branch_layers = [n_sensors * branch_in_feat] + branch_layers_list + [hidden_dim]

        full_trunk_layers = [trunk_in_feat] + trunk_layers
        model = DeepONet(
            branch_layers = branch_layers,
            trunk_layers  = full_trunk_layers,
            split_branch  = split_branch,
            split_trunk   = split_trunk,
            stacked       = stacked_do,
            output_dim    = num_outputs,
            activation    = activation,
            bias          = bias,
        )
    else:
        # Separable DeepONet
        if branch_cnn:
            raw_cnn_blocks = get_val('branch_cnn_blocks', [])
            cnn_blocks: List[List[Any]] = [list(b) for b in raw_cnn_blocks]
            if split_branch:
                cnn_blocks.append([hidden_dim * num_outputs, 'None'])
            else:
                cnn_blocks.append([hidden_dim, 'None'])

            in_channels: int = get_val('branch_cnn_input_channels', 1)
            model = SeparableBranchCNNDeepONet(
                cnn_branch_blocks = cnn_blocks,
                trunk_layers      = [1] + trunk_layers,
                in_channels       = in_channels,
                n_trunk_vars      = trunk_in_feat,
                r                 = r,
                split_branch      = split_branch,
                split_trunk       = split_trunk,
                stacked           = stacked_do,
                output_dim        = num_outputs,
                deg_grad          = deg_grad,
                activation        = activation,
                bias              = bias,
            )
        else:
            if split_branch:
                branch_layers = [n_sensors * branch_in_feat] + branch_layers_list + [num_outputs * hidden_dim]
            else:
                branch_layers = [n_sensors * branch_in_feat] + branch_layers_list + [hidden_dim]

            model = SeparableDeepONet(
                branch_layers = branch_layers,
                trunk_layers  = [1] + trunk_layers,
                n_trunk_vars  = trunk_in_feat,
                r             = r,
                split_branch  = split_branch,
                split_trunk   = split_trunk,
                stacked       = stacked_do,
                output_dim    = num_outputs,
                deg_grad      = deg_grad,
                activation    = activation,
                bias          = bias,
            )

    # Count parameters
    total_params: int = sum(p.numel() for p in model.parameters() if p.requires_grad)
    set_val('total_params', total_params)

    print("=" * 45)
    print(f"  Model:        {model.__class__.__name__}")
    print(f"  Separable:    {separable}")
    print(f"  Output Dim:   {num_outputs}")
    print(f"  Total Params: {total_params:,}")
    print("=" * 45)

    return config, model


##################################################
def main() -> None:
    return None


if __name__ == '__main__':
    main()
