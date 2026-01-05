##################################################
# Import libraries
##################################################
import  sys
from    typing                  import  Callable
from    pathlib                 import  Path
from    copy                    import  deepcopy

from    time                    import  time

import  torch
import  torch.optim             as      optim
from    deep_numerical.utils    import  compute_grad, count_parameters

sys.path.append('..')
from    models                  import  *
from    utils                   import  *




##################################################
# Configure the hyperparameters and the initial condition
##################################################
parser = TrainParser()

DEVICE = torch.device(f'cuda:{parser.cuda_index}')
torch.set_default_device(DEVICE)
SEED: int = parser.seed
torch.manual_seed(SEED)


DIMENSION:      int     = parser.dimension
MAX_T:          float   = parser.max_t
MAX_V:          float   = parser.max_v
SAMPLE_T:       int     = parser.sample_t
SAMPLE_V:       int     = parser.sample_v
SAMPLE_V_INIT:  int     = parser.sample_v_init


VHS_COEFF:      float       = parser.vhs_coeff
VHS_EXPONENT:   float       = parser.vhs_exponent
DENSITY:        float       = parser.density
INIT_TYPE:      str         = parser.init_type


IS_EXACT:       bool    = (parser.surrogate is False)
IS_TIME_FIXED:  bool    = (parser.random_time is False)
DEPTH:      int     = parser.depth
WIDTH:      int     = parser.width
SOFTPLUS:   float   = parser.softplus


LEARNING_RATE:  float   = parser.learning_rate
NUM_EPOCHS:     int     = parser.num_epochs
NUM_ITERATIONS: int     = parser.num_iterations
PERIOD_SAVE:    int     = parser.period_save
    

_prefix = 'pinn' if IS_EXACT else 'oppinn'
_result_root = 'result__fixed_t' if IS_TIME_FIXED else 'result__random_t'
path_base = Path().cwd() / \
    f'{_result_root}/{_prefix}{DIMENSION}D__vhs__coeff{VHS_COEFF:.2f}_exp{VHS_EXPONENT:.2f}__init_type_{INIT_TYPE}__seed{SEED}'
path_checkpoint = path_base / "checkpoints/"
path_base.mkdir(exist_ok=True, parents=True)
path_checkpoint.mkdir(exist_ok=True, parents=True)
sys.stdout = open(path_base / "train_history.log", 'w')


initial_condition: Callable[[torch.Tensor], torch.Tensor]
CENTER_1        = parser.init_cond__centers[0].to(DEVICE)
CENTER_2        = parser.init_cond__centers[1].to(DEVICE)
STD_1           = parser.init_cond__std.to(DEVICE)
STD_2           = parser.init_cond__std.to(DEVICE)
BKW_COEFF_EXT   = parser.bkw_coeff_ext
if INIT_TYPE=='bkw':
    initial_condition = lambda _points: bkw(_points, VHS_COEFF, BKW_COEFF_EXT, DENSITY)
elif INIT_TYPE=='maxwellian':
    initial_condition = lambda _points: maxwellian(DIMENSION, _points, CENTER_1, STD_1, DENSITY)
elif INIT_TYPE=='bimaxwellian':
    initial_condition = lambda _points: bimaxwellian(DIMENSION, _points, CENTER_1, CENTER_2, STD_1, STD_2, DENSITY)
else:
    raise ValueError(f"The initial condition type '{INIT_TYPE}' is not recognized.")


# Summarize the configuration
parser.summary()




##################################################
# Configure the collision operator and the loss function
##################################################
if IS_EXACT:
    col = FPL_spectral(
        dimension   = DIMENSION,
        v_num_grid  = SAMPLE_V,
        v_max       = MAX_V,
        vhs_coeff   = VHS_COEFF,
        vhs_alpha   = VHS_EXPONENT,
        device      = DEVICE,
    )
else:
    op_D = generate_operator_D(DIMENSION, SAMPLE_V)
    op_F = generate_operator_F(DIMENSION, SAMPLE_V)
    op_D.load_state_dict(torch.load(parser.path_D, map_location=DEVICE, weights_only=False)['state_dict'])
    op_F.load_state_dict(torch.load(parser.path_F, map_location=DEVICE, weights_only=False)['state_dict'])
    col = NeuralCollisionOperator(
        dimension   = DIMENSION,
        resolution  = SAMPLE_V,
        v_max       = MAX_V,
        op_D        = op_D,
        op_F        = op_F,
        coeff       = VHS_COEFF,
    )

def compute_loss(
        model:          torch.nn.Module,
        points_res:     torch.Tensor,
        points_init:    torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
    """Computes the loss for the PINN model by computing the predicted values and their derivatives."""
    # The FPL equation
    points_res  = torch.autograd.Variable(points_res, requires_grad=True)
    pred_int    = model.forward(points_res)
    pred_int_grad   = compute_grad(pred_int, points_res, True)
    pred_int_t      = pred_int_grad[:, [0]]
    pred_int_col: torch.Tensor
    if IS_EXACT:    pred_int_col = col.forward(pred_int)
    else:           pred_int_col = col.forward(pred_int, pred_int_grad[:, 1:])
    if pred_int_t.shape != pred_int_col.shape:
        raise ValueError(f"The temporal derivative is of shape {list(pred_int_t.shape)}, but the collision term is of shape {list(pred_int_col.shape)}.")
    
    # The initial condition
    pred_0      = model.forward(points_init)
    value_ini   = initial_condition(points_init)
    
    # Compute and return the loss terms
    residual_loss = rmse_error(pred_int_t, pred_int_col)
    initial_loss  = rmse_error(pred_0, value_ini)
    return residual_loss, initial_loss




##################################################
# Instantiate the model, the grid generator, the optimizer, and the scheduler
##################################################
model_f = PINN_FPL(
    dimension   = DIMENSION,
    depth       = DEPTH,
    width       = WIDTH,
    softplus    = SOFTPLUS,
    device      = DEVICE,
)
grid_generator = GridGenerator(DIMENSION, SAMPLE_T, MAX_T, SAMPLE_V, MAX_V, SAMPLE_V_INIT)
optimizer = optim.Adam(model_f.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=80, gamma=0.8)

print(model_f)
print(f"There are {count_parameters(model_f)} parameters.", flush=True)




##################################################
# Train the model
##################################################
logs: dict[str, list[float] | int] = {
    'loss_total':       [],
    'loss_residual':    [],
    'loss_initial':     [],
    'train_time':       [],
    'num_epochs':       0,
}

best_model: torch.nn.Module | None  = None
best_loss:  float | None            = None
best_epoch: int | None              = None
total_train_time = 0.0
for epoch in range(1, NUM_EPOCHS+1):
    losses          = AverageMeter()
    losses_residual = AverageMeter()
    losses_initial  = AverageMeter()
    model_f.train()
    
    _start_time = time()
    for iter_idx in range(NUM_ITERATIONS):
        points_res  = grid_generator.sample_tv(IS_TIME_FIXED)
        points_init = grid_generator.t0v
        loss_ge, loss_ini = compute_loss(model_f, points_res, points_init)
        loss: torch.Tensor
        if loss_ini.item()>3e-3:    loss = loss_ini
        else:                       loss = loss_ge+5*loss_ini
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.update(loss.item())
        losses_residual.update(loss_ge.item(), points_res.size(0))
        losses_initial.update(loss_ini.item(), points_init.size(0))
    scheduler.step()
    logs['train_time'].append(time()-_start_time)
    logs['loss_total'].append(losses.mean)
    logs['loss_residual'].append(losses_residual.mean)
    logs['loss_initial'].append(losses_initial.mean)
    logs['num_epochs'] = epoch
    total_train_time += logs['train_time'][-1]
    
    # Print logs
    t0v = grid_generator.t0v
    initial_error = rel_error(model_f(t0v).detach(), initial_condition(t0v))
    print(
        f"[Epoch {epoch:05d}/{NUM_EPOCHS}]",
        f"* Total loss:             {logs['loss_total'][-1]:.4e}",
        f"* Residual loss:          {logs['loss_residual'][-1]:.4e}",
        f"* Initial loss:           {logs['loss_initial'][-1]:.4e}",
        f"* Initial relative error: {initial_error:.4e}",
        "",
        sep='\n',
        flush=True,
    )
    if best_loss is None or logs['loss_total'][-1]<best_loss:
        best_model = deepcopy(model_f)
        best_loss = logs['loss_total'][-1]
        best_epoch = epoch
    if epoch%PERIOD_SAVE==0 or epoch==1 or epoch==NUM_EPOCHS:
        torch.save({'state_dict': best_model.state_dict(), 'best_epoch': best_epoch}, path_checkpoint/f"params__epoch{epoch:05d}.pth")
        torch.save(logs, path_checkpoint/f"train_history.pth")


best_model.cpu()
best_model.eval()
torch.save(
    {
        'state_dict': best_model.state_dict(),
        'best_epoch': best_epoch,
    },
    path_base/f"best_model.pth",
)
torch.save(logs, path_base/f"train_history.pth")
print(f"Training completed. (Elapsed {sec_to_hms(int(total_train_time))} for training {NUM_EPOCHS} epochs)", flush=True)




##################################################
##################################################
# End of file