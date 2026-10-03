import  torch


__all__: list[str] = ['bkw', 'maxwellian', 'bimaxwellian', 'perturbed_maxwellian']


##################################################
##################################################
def bkw(
        points:     torch.Tensor,
        kernel:     float,  # Determines `coeff_int`
        coeff_ext:  float,
        density:    float = 0.2,
    ) -> torch.Tensor:
    t = points[..., [0]]
    v = points[..., 1:]
    dim = v.shape[-1]
    coeff_int = 2*(dim-1)*density*kernel
    
    speed_sq = torch.sum(v**2, dim=-1, keepdim=True)
    K_t: torch.Tensor = 1 - coeff_ext * torch.exp(-coeff_int * t)
    
    _part1: torch.Tensor = density * torch.pow(2*torch.pi*K_t, -dim/2)
    _part2: torch.Tensor = torch.exp(-speed_sq / (2*K_t))
    _part3: torch.Tensor = ((dim+2)/2) - ((dim+speed_sq)/2)/K_t + (speed_sq/2)/(K_t**2)
    
    ret = _part1 * _part2 * _part3
    return ret


def maxwellian(
        dim:        int,
        points:     torch.Tensor,
        center:     torch.Tensor,
        sigma:      float,
        density:    float,
    ) -> torch.Tensor:
    if not (points.size(-1) in (dim, 1+dim)):
        raise ValueError(f"'points' should be a 2D tensor with size(1)={dim}, but got {list(points.shape)}.")
    if points.size(-1)==1+dim:
        points = points[:, 1:]  # Discard the time dimension if exists
    if isinstance(sigma, float):  sigma = torch.tensor([sigma], device=points.device)
    
    _cfg = {'dim': -1, 'keepdim': True}
    _dim = points.size(-1)
    mode = torch.exp(
        -(points-center).pow(2).sum(**_cfg) / \
        (2*(sigma**2))
    ) / (2*torch.pi*(sigma**2))**(0.5*_dim)
    
    return density*mode


def bimaxwellian(
        dim:        int,
        points:     torch.Tensor,
        center_1:   torch.Tensor,
        center_2:   torch.Tensor,
        sigma_1:    float,
        sigma_2:    float,
        density:    float,
    ) -> torch.Tensor:
    if not (points.size(-1) in (dim, 1+dim)):
        raise ValueError(f"'points' should be a 2D tensor with size(1)={dim}, but got {list(points.shape)}.")
    if points.size(-1)==1+dim:
        points = points[:, 1:]  # Discard the time dimension if exists
    if isinstance(sigma_1, float):  sigma_1 = torch.tensor([sigma_1], device=points.device)
    if isinstance(sigma_2, float):  sigma_2 = torch.tensor([sigma_2], device=points.device)
    
    _cfg = {'dim': -1, 'keepdim': True}
    _dim = points.size(-1)
    mode_1 = torch.exp(
        -(points-center_1).pow(2).sum(**_cfg) / \
        (2*(sigma_1**2))
    ) / (2*torch.pi*sigma_1.pow(2)).pow(0.5*_dim)
    mode_2 = torch.exp(
        -(points-center_2).pow(2).sum(**_cfg) / \
        (2*(sigma_2**2))
    ) / (2*torch.pi*sigma_2.pow(2)).pow(0.5*_dim)
    
    return density*((mode_1+mode_2)/2)


def perturbed_maxwellian(
        dim:            int,
        points:         torch.Tensor,
        center:         torch.Tensor,
        sigma:          float,
        density:        float,
        perturbation:   tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
    p0, p1, p2 = perturbation
    if any(not isinstance(p, torch.Tensor) for p in perturbation):
        raise ValueError(f"All elements of 'perturbation' should be 'torch.Tensor', but got {[type(p) for p in perturbation]}.")
    if p0.numel()!=1:
        raise ValueError(f"The first element of 'perturbation' should be a single scalar tensor, but got {p0.numel()} entries.")
    if p1.numel()!=dim:
        raise ValueError(f"The second element of 'perturbation' should be a tensor with {dim} elements, but got {p1.numel()} entries.")
    if p2.numel()!=dim**2:
        raise ValueError(f"The third element of 'perturbation' should be a tensor with {dim**2} elements, but got {p2.numel()} entries.")
    p0 = p0.reshape((1,))
    p1 = p1.reshape((dim,))
    p2 = p2.reshape((dim, dim))
    
    # sup_v = points.abs().max()
    # if p0.norm(1) + p1.norm(1)*sup_v + p2.norm(1)*(sup_v**2) > 1.0:
    #     raise ValueError("The perturbation is too large and may lead to negative values in the distribution.")
    base = maxwellian(dim, points, center, sigma, 1.0)
    perturb = (
        p0 + \
        (points * p1).sum(dim=-1, keepdim=True) + \
        torch.einsum('...i,ij,...j->...', points, p2, points).unsqueeze(-1)
    )
    return density*base*(1+perturb)


##################################################
##################################################
# End of file