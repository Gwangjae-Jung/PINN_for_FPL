from   typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import torch
import torch.nn as nn

from   .autograd import hvp_fwdfwd


__all__: List[str] = [
    'mse',
    'mse_single',
    'relative_l2_error',
    'apply_net',
    'apply_net_sep',
    'hvp_fwdfwd',
]


##################################################
def mse(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """Computes the Mean Squared Error (MSE) between true values and predictions.

    ## Description
    Calculates $\\frac{1}{N}\\sum_{i=1}^N (y_{\\text{true}, i} - y_{\\text{pred}, i})^2$.

    Arguments:
        `y_true` (`torch.Tensor`): Ground truth target tensor.
        `y_pred` (`torch.Tensor`): Predicted output tensor from neural network.

    Returns:
        `torch.Tensor`: Scalar mean squared error tensor.
    """
    return torch.mean(torch.square(y_true - y_pred))


def mse_single(y_pred: torch.Tensor) -> torch.Tensor:
    """Computes the mean squared magnitude of a tensor.

    ## Description
    Calculates $\\frac{1}{N}\\sum_{i=1}^N (y_{\\text{pred}, i})^2$.
    Typically used to evaluate residual losses of homogeneous differential equations (e.g., $\\mathcal{N}(s) = 0$).

    Arguments:
        `y_pred` (`torch.Tensor`): Residual or error tensor.

    Returns:
        `torch.Tensor`: Scalar mean squared value.
    """
    return torch.mean(torch.square(y_pred))


def relative_l2_error(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """Computes the relative $L_2$ error norm.

    ## Description
    Calculates $\\frac{\\|y_{\\text{true}} - y_{\\text{pred}}\\|_2}{\\|y_{\\text{true}}\\|_2}$.

    Arguments:
        `y_true` (`torch.Tensor`): Ground truth reference tensor.
        `y_pred` (`torch.Tensor`): Predicted output tensor.

    Returns:
        `torch.Tensor`: Scalar relative $L_2$ error.
    """
    diff_norm: torch.Tensor = torch.linalg.norm(y_true.reshape(-1) - y_pred.reshape(-1))
    true_norm: torch.Tensor = torch.linalg.norm(y_true.reshape(-1))
    return diff_norm / (true_norm + 1e-12)


def apply_net(
    model:        nn.Module,
    branch_input: torch.Tensor,
    *trunk_in:    torch.Tensor,
) -> torch.Tensor:
    """Forward pass wrapper for conventional DeepONet with multiple trunk coordinate inputs.

    ## Description
    Stacks 1D coordinate inputs into a single coordinate tensor along the last axis
    and evaluates the standard DeepONet module.
    If the output channel dimension is 1, it squeezes the last dimension for convenient downstream computation.

    Arguments:
        `model` (`nn.Module`): Conventional `DeepONet` module instance.
        `branch_input` (`torch.Tensor`): Branch input tensor evaluated at sensor locations.
        `*trunk_in` (`torch.Tensor`): Coordinate tensors to stack and pass to the trunk network.

    Returns:
        `torch.Tensor`: Output evaluation tensor.
    """
    trunk_input: torch.Tensor
    if len(trunk_in) == 1:
        trunk_input = trunk_in[0]
    else:
        # Stack coordinates along the last dimension
        reshaped_trunks = [t.reshape(-1, 1) if t.ndim == 1 else t for t in trunk_in]
        trunk_input = torch.cat(reshaped_trunks, dim=-1)

    out: torch.Tensor = model(branch_input, trunk_input)

    # Squeeze channel dimension if 1 and output has rank > 1
    if out.shape[-1] == 1 and out.ndim > 1:
        out = out.squeeze(-1)
    return out


def apply_net_sep(
    model:        nn.Module,
    branch_input: torch.Tensor,
    *trunk_in:    torch.Tensor,
) -> torch.Tensor:
    """Forward pass wrapper for Separable DeepONet models.

    ## Description
    Invokes the forward method of `SeparableDeepONet` or `SeparableBranchCNNDeepONet`
    with factorized 1D coordinate inputs.

    Arguments:
        `model` (`nn.Module`): `SeparableDeepONet` or `SeparableBranchCNNDeepONet` instance.
        `branch_input` (`torch.Tensor`): Input functions or fields for branch network.
        `*trunk_in` (`torch.Tensor`): Factorized 1D coordinate tensors.

    Returns:
        `torch.Tensor`: Output prediction tensor on the multidimensional coordinate lattice.
    """
    out: torch.Tensor = model(branch_input, *trunk_in)
    return out


##################################################
def main() -> None:
    return None


if __name__ == '__main__':
    main()
