from   typing            import Any, Callable, List, Optional, Sequence, Tuple, Union
from   typing_extensions import TypeAlias
import torch
from   torch.func        import jacfwd, jacrev, jvp, vmap


__all__:           List[str] = ['jacobian', 'hessian', 'derivatives', 'hvp_fwdfwd', 'compute_grad']
FuncType:        TypeAlias = Callable[[torch.Tensor], torch.Tensor]
FuncTypeWithAux: TypeAlias = Callable[[torch.Tensor], Tuple[torch.Tensor, Any]]


##################################################
def jacobian(func: FuncType, return_out: bool = False) -> Union[FuncType, FuncTypeWithAux]:
    """Computes the Jacobian function using forward-mode automatic differentiation.

    ## Description
    Returns a vectorized function (`vmap`-wrapped) that computes the Jacobian of `func` using `torch.func.jacfwd`.
    If `return_out` is `True`, the returned function returns both the Jacobian and the original evaluation.

    Arguments:
        `func` (`FuncType`): A function mapping an input tensor to an output tensor.
        `return_out` (`bool`, default: `False`): Flag determining whether to also return the evaluated function output.

    Returns:
        `Union[FuncType, FuncTypeWithAux]`: Vectorized function computing the Jacobian tensor.
    """
    if return_out:
        def modified_func(pts: torch.Tensor, **kwargs: Any) -> Tuple[torch.Tensor, torch.Tensor]:
            out: torch.Tensor = func(pts, **kwargs)
            return out, out
        return vmap(jacfwd(modified_func, has_aux=True))
    else:
        return vmap(jacfwd(func))


def hessian(func: FuncType, return_out: bool = False) -> Union[FuncType, FuncTypeWithAux]:
    """Computes the Hessian function using forward-mode automatic differentiation.

    ## Description
    Returns a vectorized function (`vmap`-wrapped) that computes the Hessian of `func` using forward-over-forward mode autodiff.
    If `return_out` is `True`, the returned function returns both the Hessian and intermediate evaluation output.

    Arguments:
        `func` (`FuncType`): A function mapping an input tensor to an output tensor.
        `return_out` (`bool`, default: `False`): Flag determining whether to also return intermediate output.

    Returns:
        `Union[FuncType, FuncTypeWithAux]`: Vectorized function computing the Hessian tensor.
    """
    if return_out:
        def modified_func(pts: torch.Tensor, **kwargs: Any) -> Tuple[torch.Tensor, torch.Tensor]:
            out: torch.Tensor = func(pts, **kwargs)
            return out, out
        return vmap(jacfwd(jacfwd(modified_func, has_aux=True), has_aux=True))
    else:
        return vmap(jacfwd(jacfwd(func)))


def derivatives(func: FuncType, degree: int = 1) -> Callable[[torch.Tensor], List[torch.Tensor]]:
    """Computes higher-order derivatives of a function using forward-mode automatic differentiation.

    ## Description
    Returns a vectorized function (`vmap`-wrapped) that iteratively computes derivatives of `func` up to the given `degree` using `torch.func.jacfwd`.
    The returned list contains `[f(x), f'(x), f''(x), ...]`, where each entry is reshaped to have the batch dimension first.

    Arguments:
        `func` (`FuncType`): A function mapping an input point tensor to an output vector.
        `degree` (`int`, default: `1`): The maximum order of differentiation to evaluate.

    Returns:
        `Callable[[torch.Tensor], List[torch.Tensor]]`: Vectorized function returning a list of derivative tensors from order 0 to `degree`.
    """
    def modified_func(pts: torch.Tensor, **kwargs: Any) -> Tuple[torch.Tensor, List[Any]]:
        return func(pts, **kwargs), []

    def repack(_func: Any, _return_out: bool = True) -> Callable[..., Any]:
        def _repacked_func(pts: torch.Tensor, **kwargs: Any) -> Union[Tuple[torch.Tensor, List[Any]], List[Any]]:
            _out, _aux = _func(pts, **kwargs)
            _aux.append(_out)
            if _return_out:
                return _out, _aux
            else:
                return _aux
        return _repacked_func

    _vmapped: Any = modified_func
    for _ in range(1, 1 + degree):
        _vmapped = jacfwd(repack(_vmapped), has_aux=True)

    vmap_eval = vmap(repack(_vmapped, False))

    def evaluate_derivatives(pts: torch.Tensor) -> List[torch.Tensor]:
        raw_outputs: List[torch.Tensor] = vmap_eval(pts)
        results:     List[torch.Tensor] = []
        batch_size:  int                = pts.shape[0]
        for out in raw_outputs:
            results.append(out.reshape(batch_size, -1))
        return results

    return evaluate_derivatives


def hvp_fwdfwd(
    func:           Callable[[torch.Tensor], torch.Tensor],
    primals:        Tuple[torch.Tensor, ...],
    tangents:       Tuple[torch.Tensor, ...],
    return_primals: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Computes forward-over-forward Hessian-vector product using `torch.func.jvp`.

    ## Description
    Computes the directional derivative of the Jacobian-vector product using nested forward-mode autodiff (`jvp`).
    This operation is mathematically equivalent to $H \\cdot v$, where $H$ is the Hessian matrix and $v$ is the tangent vector.

    Arguments:
        `func` (`Callable[[torch.Tensor], torch.Tensor]`): The scalar or vector-valued function to differentiate.
        `primals` (`Tuple[torch.Tensor, ...]`): The primal evaluation point tuple.
        `tangents` (`Tuple[torch.Tensor, ...]`): The tangent vector tuple corresponding to the directional derivative.
        `return_primals` (`bool`, default: `False`): Flag indicating whether to return the function value along with the Hessian product.

    Returns:
        `Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]`: Evaluated Hessian-vector product, or tuple of primal output and Hessian product.
    """
    g = lambda p: jvp(func, (p,), tangents)[1]
    primals_out, tangents_out = jvp(g, primals, tangents)
    if return_primals:
        return primals_out, tangents_out
    return tangents_out


def compute_grad(
    outputs:      torch.Tensor,
    inputs:       torch.Tensor,
    create_graph: bool           = True,
    retain_graph: Optional[bool] = None,
) -> torch.Tensor:
    """Computes the gradient of `outputs` with respect to `inputs` via reverse-mode automatic differentiation.

    ## Description
    Calls `torch.autograd.grad` with `grad_outputs` initialized to all ones.
    If `inputs` does not contribute to `outputs`, a tensor of zeros matching `inputs.shape` is returned.

    Arguments:
        `outputs` (`torch.Tensor`): Dependent variable tensor whose gradients are to be computed.
        `inputs` (`torch.Tensor`): Independent variable tensor with respect to which gradients are taken.
        `create_graph` (`bool`, default: `True`): Flag determining whether the computational graph for derivatives should be created.
        `retain_graph` (`Optional[bool]`, default: `None`): Flag determining whether to retain graph; defaults to `create_graph` if `None`.

    Returns:
        `torch.Tensor`: Evaluated gradient tensor.
    """
    if retain_graph is None:
        retain_graph = create_graph

    grad_val: Optional[torch.Tensor] = torch.autograd.grad(
        outputs      = outputs,
        inputs       = inputs,
        grad_outputs = torch.ones_like(outputs),
        create_graph = create_graph,
        retain_graph = retain_graph,
        allow_unused = True,
    )[0]

    if grad_val is None:
        return torch.zeros_like(inputs)
    return grad_val


##################################################
def main() -> None:
    return None


if __name__ == '__main__':
    main()
