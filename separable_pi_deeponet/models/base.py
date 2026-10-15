from   typing import Any, List
import torch


__all__: List[str] = ['BaseModule']


##################################################
class BaseModule(torch.nn.Module):
    """Base neural network module with parameter counting, architecture summaries, and validation hooks.

    ## Description
    Serves as the common foundation for all DeepONet and Separable DeepONet architectures in this package.
    Provides diagnostic methods such as parameter count reporting, structural string representation, and configuration checks.
    """

    def __init__(self) -> None:
        """Initializes the base module."""
        super().__init__()
        return None

    def count_parameters(self) -> int:
        """Counts the total number of trainable parameters in the module.

        ## Description
        Iterates over all parameters in the module and sums their elements.
        Complex parameters are counted as two real numbers.

        Returns:
            `int`: The total number of trainable parameters.
        """
        cnt: int = 0
        for p in self.parameters():
            if p.requires_grad:
                c: int = 2 if p.is_complex() else 1
                cnt += c * p.numel()
        return cnt

    def check_arguments(self, **kwargs: Any) -> None:
        """Validates layer and architectural arguments.

        ## Description
        Hook for subclasses to validate architectural specifications before constructing submodules.

        Arguments:
            `**kwargs` (`Any`): Keyword arguments to check.

        Returns:
            This method returns no values.
        """
        return None

    def __str__(self) -> str:
        """Returns a formatted summary string of subnetworks and parameters.

        ## Description
        Generates an ASCII overview table listing subnetworks, parameters, shapes, types, and parameter count.

        Returns:
            `str`: Formatted overview string.
        """
        msg:         List[str] = []
        __half_line: str       = '=' * 30
        _front:      str       = ''.join((__half_line, f'< {self.__class__.__name__} >', __half_line))
        _line:       str       = '-' * len(_front)
        _back:       str       = '=' * len(_front)

        msg.append(_front)
        msg.append("[ Subnetworks ]\n")
        for name, md in self.named_children():
            msg.append(f"* {name}")
            msg.append(str(md))
            msg.append('')

        msg.append(_line)
        msg.append("[ Parameters ]")
        named_params = self.named_parameters(recurse=False)
        for name, p in named_params:
            msg.append(f"( {name} )")
            msg.append(f"- Shape:       {list(p.shape)}")
            msg.append(f"- Data type:   {p.dtype}")

        msg.append(_line)
        msg.append(f"Number of parameters: {self.count_parameters()}")
        msg.append(_back)

        return '\n'.join(msg)


##################################################
def main() -> None:
    return None


if __name__ == '__main__':
    main()
