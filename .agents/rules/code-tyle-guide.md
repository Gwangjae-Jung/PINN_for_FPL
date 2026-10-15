---
trigger: always_on
---

## 1. Python coding style
Below are some guidelines when you create Python scripts.
1. If possible, vertically align the words in sentences of similar work in the same paragraph. Refer to <a href="#example01">[Example 01]</a> for example.
2. The type of each object and argumet of functions should be annotated.
    - For complex dictionaries where the multiple types are allowed for the key and/or values, it is recommended to use `typing.TypedDict`.
    - Avoid using `Any` or `object`, and annotate types precisely.
    - If the types of variables cannot be inferred from docstrings (for example, when the return type of a function is annotated as `Any` or `object` rather than a specific (collection of) datatypes), you should explicitly annotate the types of the variables.
3. Docstrings should be written for each function, class, and method of classes (possible exception: The function `main()`, the unique function called in `if __name__=="__main__":` phrase). The type of each argument of each function(method) should be given in parentheses with enclosed by backticks. Refer to <a href="#example01">[Example 01]</a> for example.
4. Use `if __name__=="__main__":` phrase if the script is constructed to be run. This need not be applied to the scripts to be imported.
5. When importing libraries, import libraries in the following order: {Python built-in libraries}, {Third-party libraries available in `pip` or `conda`}, {*Installed* custom libraries available in GitHub}, {Submodules in the project}.
6. Semantic separation can be marked using a comment line of fifty `'#'`s. See <a href="#example01">[Example 01]</a>.
7. The implementation of each function(method) should end with a `return` phrase, even if the return type is `None`; in this case, the implementation should end with the single line `return`, and the `Returns:` section in the docstring should be written as in <a href="#example01">[Example 01]</a>.
8. Avoid hardcoding; define constant variables on the top of the scripts.

## Appendix 1
```python
# run_experiment.py (This file is constructed to be run, rather than to be imported by another script.)
import os
from dotenv import load_dotenv
from pathlib import Path
from typing import TypedDict, Callable, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset
from matplotlib import pyplot as plt

is_env_loaded = load_dotenv()
if not is_env_loaded:
    print("No '.env' file is loaded.")


##################################################
class TrainInfo(TypedDict):
    """A custom class for saving information about model training.
    """
    num_epochs: int
    """The number of training epochs."""
    learning_rate: float
    """The base value of the learning rate."""


def _function_template(arg1: int, arg2: str, arg3: Callable[[int, int], int]) -> int:
    """{Brief explanation of the function}

    ## Description                          // If detail explanation is not necessary,
    {Detail explanation of the function}    // this section can be omitted.

    Arguments:
        `arg1` (`int`): {Explanation of `arg1`}
        `arg2` (`str`): {Explanation of `arg2`}
        `arg3` (`Callable[[int, int], int]`): {Explanation of `arg3`}
    
    Returns:
        `{return type}`: {Explanation of the return value}
    """
    pass


def load_dataset(path: Union[str, Path], verbose: bool=False) -> TensorDataset:
    """Loads the dataset from `path` which consists of the pair of input and output.

    Arguments:
        `path` (`Union[str, Path]`): The path of the dataset to be loaded.
        `verbose` (`bool`, default: `False`): The flag which determines to print information regarding the data loading.
    
    Returns:
        `torch.utils.data.TensorDataset`: The dataset which consists of the input and output.
    """
    with open(path, "r") as file:
        x: torch.Tensor = file['input']   # Explicit annotation of the variables.
        y: torch.Tensor = file['output']  # Note that the type of `file[k]` is not annotated for each key `k` of `file`.
    return TensorDataset(x, y)


def report_dataset(dataset: TensorDataset) -> None:
    """Reports information regarding `dataset`.

    ## Description
    This function prints several information on `dataset`.
    To be specific, this function returns the size of the dataset, the shape of each sample with the batch dimension.

    Arguments:
        `dataset` (`torch.utils.data.TensorDataset`): The dataset to be reported.

    Returns:
        This function returns no values.
    """
    print(f"Size of the dataset >>>  {len(dataset)}")
    print(f"Shape of each sample >>> {dataset[[0]].shape}") 
    return


##################################################
def main() -> None:
    dataset = load_dataset("data/dataset_01.pth")
    report_dataset(dataset)
    return


if __name__ == "__main__":
    main()

```