from   typing   import Sequence, Union
from   datetime import timedelta
import torch
import torch.nn as nn


__all__: list[str] = ['sec_to_hms', 'count_parameters']


##################################################
def sec_to_hms(seconds: float) -> str:
    """Convert seconds to a formatted HH:MM:SS string.

    ## Description
    Converts a floating-point duration in seconds into a human-readable string
    formatted as `H:MM:SS` (using Python's `datetime.timedelta`).

    ## Arguments
    `seconds` (`float`): Duration in seconds.

    ## Returns
    `str`: Formatted time string of the form `H:MM:SS`.
    """
    return str(timedelta(seconds=int(seconds)))


##################################################
def count_parameters(
        models:         Union[nn.Module, Sequence[nn.Module]],
        complex_as_two: bool = True,
    ) -> Union[int, list[int]]:
    """Count the total number of learnable parameters in one or more models.

    ## Description
    Iterates over the parameters of each provided `nn.Module` and sums those that
    require gradients. Complex-valued parameters can optionally be counted as two
    real values each.

    ## Arguments
    `models` (`Union[nn.Module, Sequence[nn.Module]]`): A single PyTorch model or a
    sequence of models whose learnable parameters are to be counted.
    `complex_as_two` (`bool`, default: `True`): If `True`, each complex-valued
    parameter element is counted as two real scalars.

    ## Returns
    `Union[int, list[int]]`: Total learnable parameter count for a single model,
    or a list of per-model counts when a sequence is provided.
    """
    is_single:  bool                  = isinstance(models, nn.Module)
    model_list: Sequence[nn.Module]   = [models] if is_single else models

    counts: list[int] = []
    for model in model_list:
        total: int = 0
        for param in model.parameters():
            if param.requires_grad:
                mult: int = 2 if (complex_as_two and param.is_complex()) else 1
                total += param.numel() * mult
        counts.append(total)

    if is_single:
        return counts[0]
    return counts


##################################################
##################################################
# End of file