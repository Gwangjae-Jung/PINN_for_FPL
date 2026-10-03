from   typing   import Sequence, Union
from   datetime import timedelta
import torch
import torch.nn as nn


__all__: list[str] = ['sec_to_hms', 'count_parameters']


##################################################
def sec_to_hms(seconds: float) -> str:
    """## Convert seconds to formatted HH:MM:SS string

## Description
Converts a floating-point duration in seconds into a human-readable string formatted as `H:MM:SS`.

## Arguments
`seconds` (`float`): Duration in seconds.

## Returns
`str`: Formatted time string.
"""
    return str(timedelta(seconds=int(seconds)))


def count_parameters(
        models:         Union[nn.Module, Sequence[nn.Module]],
        complex_as_two: bool = True,
    ) -> Union[int, list[int]]:
    """## Count learnable model parameters

## Description
Counts the total number of learnable parameters in one or more PyTorch `nn.Module` instances.

## Arguments
`models` (`Union[nn.Module, Sequence[nn.Module]]`): PyTorch model or sequence of models.
`complex_as_two` (`bool`, default: `True`): If `True`, complex parameters are counted as two real values.

## Returns
`Union[int, list[int]]`: Number of learnable parameters, or a list of counts if multiple models were provided.
"""
    is_single: bool = isinstance(models, nn.Module)
    model_list: Sequence[nn.Module] = [models] if is_single else models

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