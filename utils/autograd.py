from    typing      import  Optional
import  torch


__all__: list[str] = ['compute_grad']


def compute_grad(
        outputs:        torch.Tensor,
        inputs:         torch.Tensor,
        create_graph:   bool            = True,
        retain_graph:   Optional[bool]  = None,
    ) -> torch.Tensor:
    """## Autograd for computing gradients
    
    Arguments:
        `outputs` (`torch.Tensor`): A 1-dimensional tensor, which acts as an array of the values of a function of `inputs`. This function aims at computing the gradient of `outputs` at `inputs`.
        
        `inputs` (`torch.Tensor`): A tensor object at which the gradient of `outputs` shall be computed.

        `create_graph` (`bool`, default: `True`): See Appendix below.
        
        `retain_graph` (`bool`, default: `None`): See Appendix below. When this parameter is not initialized, it is initialized to `create_graph` by default.
    
    Returns:
        This function returns the gradient of `outputs` at `inputs`.
        
    -----
    ### Remark
    1. (Dimensionality)
        `outputs` is required to be a tensor of dimension 1.
    2. (Slicing)
        Since generates a new tensor, slicing `outputs` does not matter in back-propagation, while slicing `inputs` generates a tensor which is not connected with `outputs` in the computational graph of `outputs`.
    
    -----
    ### Examples

    Example 1.
    >>> x = torch.tensor([1, 2, 3], dtype = torch.float, requires_grad = True)
    >>> w = (x ** 2).sum()
    >>> compute_grad(outputs = w, inputs = x, create_graph = False)
    tensor([2., 4., 6.], grad_fn=<...>)

    Example 2.
    >>> x = torch.tensor([4, 0], dtype = torch.float, requires_grad = True)
    >>> u = 2 * x[0] + torch.exp(1 + x[1] ** 2 )
    >>> u_grad = compute_grad(u, x, create_graph = True)
    >>> u_grad
    tensor([2., 0.], grad_fn=<...>)
    >>> u_x = u_grad[0]
    >>> u_xy = compute_grad(u_x, x, create_graph = False)[1]
    >>> u_xy
    tensor([0.])
    >>> u_xy.requires_grad
    False

    Example 3.
    >>> To be added.

    -----
    ### Appendix. Some parameters of `torch.autograd.grad()`
    Here, `R` is the set of the real numbers.
    Let `U` be a nonempty open subset of `R^n` and `f: U --> R^k` be a map with a computational graph.

    1. (`outputs`, `inputs`, and `grad_outputs` (`torch.Tensor`))
    As the names indicate, `inputs` is a tensor of points in `U` and `outputs` is a tensor of values of `f` at each point in `inputs`.
    `torch.autograd.grad()` first computes the gradients (of `outputs` at the points listed in `inputs`), then does the vector-Jacobian multiplication.
    To compute the partial derivatives for each member of the input batch, `grad_outputs` has to be manually set `torch.oneslike(outputs)`, which is observed by the author.

    2-1. (`create_graph` (`bool`, default: `True`))
    This boolean parameter determines whether `torch.autograd.grad()` creates the computational graph for the derivative, which shall be generally used to compute derivatives of higher order.
        * If `True`, the computational graph for the derivative shall be constructed.
        * If `False`, the computational graph for the derivative is not constructed.

    2-2. (`retain_graph` (`bool`, default: `create_graph`)))
    This boolean parameter determines whether the computational graph for computing the derivative should be preserved.
        * When `True`, the computational graph is preserved.
        * When `False`, the computational graph is deleted.
    
    3. (`allow_unused` (`bool`))
    When `inputs` does not contribute in computing `outputs` (so that the true gradient is obviously the zero vector), then `inputs` is not contained in the computational graph of `outputs`, so `torch.autograd.grad()` cannot properly compute the gradient.
        * When `True`, then `torch.autograd.grad()` returns the tuple `(None,)`.
        * When `False`, then `torch.autograd.grad()` throws a runtime error.
    """
    # Initialize `retain_graph` if it is not initialized
    if retain_graph is None:
        retain_graph = create_graph
    
    # Compute the gradient
    return_value = torch.autograd.grad(
        outputs         = outputs,
        inputs          = inputs,
        grad_outputs    = torch.ones_like(outputs),
        create_graph    = create_graph,
        retain_graph    = retain_graph,
        allow_unused    = True,
    )[0]
    
    # If no backward propagation is executed, set the gradient to the zero vector
    if return_value is None:
        return_value = torch.zeros_like(inputs)
    
    return return_value