# Import necessary libraries and modules.
from   copy       import deepcopy
from   pathlib    import Path
import sys
from   time       import time
from   typing     import Callable

import torch
import torch.optim as optim

_PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from   models import (
    FPL_spectral,
    NeuralCollisionOperator,
    PINN_FPL,
    generate_operator_D,
    generate_operator_F,
)
from   utils  import (
    AverageMeter,
    GridGenerator,
    TrainParser,
    bimaxwellian,
    bkw,
    compute_grad,
    count_parameters,
    maxwellian,
    rel_error,
    rmse_error,
    sec_to_hms,
)


##################################################
def _compute_loss(
    model:             torch.nn.Module,
    points_res:        torch.Tensor,
    points_init:       torch.Tensor,
    col:               torch.nn.Module | Callable,
    initial_condition: Callable[[torch.Tensor], torch.Tensor],
    is_exact:          bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the loss for the PINN model by evaluating PDE residuals and initial conditions.

    ## Description
    Computes the temporal derivative and collision operator output to evaluate
    the FPL equation residual, along with the error against the initial condition.

    Arguments:
        `model` (`torch.nn.Module`): The PINN model being trained.
        `points_res` (`torch.Tensor`): Residual collocation points (t, v).
        `points_init` (`torch.Tensor`): Initial condition collocation points (t=0, v).
        `col` (`torch.nn.Module | Callable`): Collision operator (spectral or neural surrogate).
        `initial_condition` (`Callable[[torch.Tensor], torch.Tensor]`): Function computing true initial values.
        `is_exact` (`bool`): True if exact spectral solver is used, False if neural surrogate is used.

    Returns:
        `tuple[torch.Tensor, torch.Tensor]`: Tuple containing (residual_loss, initial_loss).
    """
    # The FPL equation
    points_res    = torch.autograd.Variable(points_res, requires_grad=True)
    pred_int      = model.forward(points_res)
    pred_int_grad = compute_grad(pred_int, points_res, True)
    pred_int_t    = pred_int_grad[:, [0]]
    pred_int_col: torch.Tensor
    if is_exact:
        pred_int_col = col.forward(pred_int)
    else:
        pred_int_col = col.forward(pred_int, pred_int_grad[:, 1:])

    if pred_int_t.shape != pred_int_col.shape:
        raise ValueError(
            f"The temporal derivative is of shape {list(pred_int_t.shape)}, "
            f"but the collision term is of shape {list(pred_int_col.shape)}."
        )

    # The initial condition
    pred_0    = model.forward(points_init)
    value_ini = initial_condition(points_init)

    # Compute and return the loss terms
    residual_loss = rmse_error(pred_int_t, pred_int_col)
    initial_loss  = rmse_error(pred_0, value_ini)
    return residual_loss, initial_loss


def _main() -> None:
    """Execute the PINN training procedure for the Fokker-Planck-Landau equation.

    ## Description
    Initializes device, seeds, networks, collision operators, optimization loops,
    and checkpoint saving.

    Arguments:
        This function does not take arguments.

    Returns:
        This function does not return values.
    """
    # Configure the hyperparameters and the initial condition
    parser = TrainParser()

    device = torch.device(f'cuda:{parser.cuda_index}')
    torch.set_default_device(device)
    seed: int = parser.seed
    torch.manual_seed(seed)

    dimension:     int   = parser.dimension
    max_t:         float = parser.max_t
    max_v:         float = parser.max_v
    sample_t:      int   = parser.sample_t
    sample_v:      int   = parser.sample_v
    sample_v_init: int   = parser.sample_v_init

    vhs_coeff:    float = parser.vhs_coeff
    vhs_exponent: float = parser.vhs_exponent
    density:      float = parser.density
    init_type:    str   = parser.init_type

    is_exact:      bool  = (parser.surrogate is False)
    is_time_fixed: bool  = (parser.random_time is False)
    depth:         int   = parser.depth
    width:         int   = parser.width
    softplus:      float = parser.softplus

    learning_rate:  float = parser.learning_rate
    num_epochs:     int   = parser.num_epochs
    num_iterations: int   = parser.num_iterations
    period_save:    int   = parser.period_save

    prefix:      str  = 'pinn' if is_exact else 'oppinn'
    result_root: str  = 'result__fixed_t' if is_time_fixed else 'result__random_t'
    path_base:   Path = _PROJECT_ROOT / result_root / \
        f'{prefix}{dimension}D__vhs__coeff{vhs_coeff:.2f}_exp{vhs_exponent:.2f}__init_type_{init_type}__seed{seed}'
    path_checkpoint: Path = path_base / "checkpoints/"
    path_base.mkdir(exist_ok=True, parents=True)
    path_checkpoint.mkdir(exist_ok=True, parents=True)
    sys.stdout = open(path_base / "train_history.log", 'w')

    center_1:      torch.Tensor = parser.init_cond__centers[0].to(device)
    center_2:      torch.Tensor = parser.init_cond__centers[1].to(device)
    std_1:         torch.Tensor = parser.init_cond__std.to(device)
    std_2:         torch.Tensor = parser.init_cond__std.to(device)
    bkw_coeff_ext: float        = parser.bkw_coeff_ext

    initial_condition: Callable[[torch.Tensor], torch.Tensor]
    if init_type == 'bkw':
        initial_condition = lambda _points: bkw(_points, vhs_coeff, bkw_coeff_ext, density)
    elif init_type == 'maxwellian':
        initial_condition = lambda _points: maxwellian(dimension, _points, center_1, std_1, density)
    elif init_type == 'bimaxwellian':
        initial_condition = lambda _points: bimaxwellian(dimension, _points, center_1, center_2, std_1, std_2, density)
    else:
        raise ValueError(f"The initial condition type '{init_type}' is not recognized.")

    # Summarize the configuration
    parser.summary()

    # Configure the collision operator and the loss function
    col: torch.nn.Module | Callable
    if is_exact:
        col = FPL_spectral(
            dimension   = dimension,
            v_num_grid  = sample_v,
            v_max       = max_v,
            vhs_coeff   = vhs_coeff,
            vhs_alpha   = vhs_exponent,
            device      = device,
        )
    else:
        op_d = generate_operator_D(dimension, sample_v)
        op_f = generate_operator_F(dimension, sample_v)

        path_d = Path(parser.path_D)
        if not path_d.is_absolute():
            path_d = _PROJECT_ROOT / path_d
        path_f = Path(parser.path_F)
        if not path_f.is_absolute():
            path_f = _PROJECT_ROOT / path_f

        op_d.load_state_dict(torch.load(path_d, map_location=device, weights_only=False)['state_dict'])
        op_f.load_state_dict(torch.load(path_f, map_location=device, weights_only=False)['state_dict'])
        col = NeuralCollisionOperator(
            dimension   = dimension,
            resolution  = sample_v,
            v_max       = max_v,
            op_D        = op_d,
            op_F        = op_f,
            coeff       = vhs_coeff,
        )

    # Instantiate the model, the grid generator, the optimizer, and the scheduler
    model_f = PINN_FPL(
        dimension   = dimension,
        depth       = depth,
        width       = width,
        softplus    = softplus,
        device      = device,
    )
    grid_generator = GridGenerator(dimension, sample_t, max_t, sample_v, max_v, sample_v_init)
    optimizer      = optim.Adam(model_f.parameters(), lr=learning_rate)
    scheduler      = torch.optim.lr_scheduler.StepLR(optimizer, step_size=80, gamma=0.8)

    print(model_f)
    print(f"There are {count_parameters(model_f)} parameters.", flush=True)

    # Train the model
    logs: dict[str, list[float] | int] = {
        'loss_total':    [],
        'loss_residual': [],
        'loss_initial':  [],
        'train_time':    [],
        'num_epochs':    0,
    }

    best_model:       torch.nn.Module | None = None
    best_loss:        float | None           = None
    best_epoch:       int | None             = None
    total_train_time: float                  = 0.0

    for epoch in range(1, num_epochs + 1):
        losses          = AverageMeter()
        losses_residual = AverageMeter()
        losses_initial  = AverageMeter()
        model_f.train()

        start_time = time()
        for _ in range(num_iterations):
            points_res        = grid_generator.sample_tv(is_time_fixed)
            points_init       = grid_generator.t0v
            loss_ge, loss_ini = _compute_loss(model_f, points_res, points_init, col, initial_condition, is_exact)
            loss: torch.Tensor
            if loss_ini.item() > 3e-3:
                loss = loss_ini
            else:
                loss = loss_ge + 5 * loss_ini
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.update(loss.item())
            losses_residual.update(loss_ge.item(), points_res.size(0))
            losses_initial.update(loss_ini.item(), points_init.size(0))

        scheduler.step()
        logs['train_time'].append(time() - start_time)
        logs['loss_total'].append(losses.mean)
        logs['loss_residual'].append(losses_residual.mean)
        logs['loss_initial'].append(losses_initial.mean)
        logs['num_epochs'] = epoch
        total_train_time  += logs['train_time'][-1]

        # Print logs
        t0v           = grid_generator.t0v
        initial_error = rel_error(model_f(t0v).detach(), initial_condition(t0v))
        print(
            f"[Epoch {epoch:05d}/{num_epochs}]",
            f"* Total loss:             {logs['loss_total'][-1]:.4e}",
            f"* Residual loss:          {logs['loss_residual'][-1]:.4e}",
            f"* Initial loss:           {logs['loss_initial'][-1]:.4e}",
            f"* Initial relative error: {initial_error:.4e}",
            "",
            sep='\n',
            flush=True,
        )
        if best_loss is None or logs['loss_total'][-1] < best_loss:
            best_model = deepcopy(model_f)
            best_loss  = logs['loss_total'][-1]
            best_epoch = epoch
        if epoch % period_save == 0 or epoch == 1 or epoch == num_epochs:
            torch.save(
                {'state_dict': best_model.state_dict(), 'best_epoch': best_epoch},
                path_checkpoint / f"params__epoch{epoch:05d}.pth",
            )
            torch.save(logs, path_checkpoint / "train_history.pth")

    if best_model is not None:
        best_model.cpu()
        best_model.eval()
        torch.save(
            {
                'state_dict': best_model.state_dict(),
                'best_epoch': best_epoch,
            },
            path_base / "best_model.pth",
        )
    torch.save(logs, path_base / "train_history.pth")
    print(
        f"Training completed. (Elapsed {sec_to_hms(int(total_train_time))} for training {num_epochs} epochs)",
        flush=True,
    )
    return


##################################################
if __name__ == '__main__':
    _main()