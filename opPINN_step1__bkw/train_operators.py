##################################################
# Import libraries
##################################################
import  argparse
from    pathlib     import  Path
from    time        import  time
from    copy        import  deepcopy

import  torch
from    torch.utils.data    import  DataLoader

import  sys
sys.path.append('..')
from    models      import  generate_operator_D, generate_operator_F
from    helper      import  mse_loss, relative_error, Dataset_FPL
from    utils       import  sec_to_hms


parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, help='Random seed for reproducibility.')
parser.add_argument('--cuda_index', type=int, help='CUDA device index to use for training.')
parser.add_argument('--dimension', type=int, choices=[2, 3], help='Dimension of the problem (2 or 3).')
parser.add_argument('--resolution', type=int, choices=[32, 64, 128], help='Resolution of the grid.')
parser.add_argument('--density', type=float, default=0.2, help='The density of the input distribution functions.')
parser.add_argument('--operator', type=str, choices=['D', 'F'], help='Operator to train: F or D.')
parser.add_argument('--batch_size', type=int, default=100, help='Batch size for training.')
parser.add_argument('--num_epochs', type=int, default=int(5e4), help='Number of training epochs.')
parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate for the optimizer.')
parser.add_argument('--scheduler_step', type=int, default=2000, help='Step size for the learning rate scheduler.')
parser.add_argument('--scheduler_decay', type=float, default=0.9, help='Decay factor for the learning rate scheduler.')
parser.add_argument('--period_report', type=int, default=100, help='Period of reporting during training.')
parser.add_argument('--period_backup', type=int, default=1000, help='Period of backing up the model during training.')
args = parser.parse_args()



##################################################
# Configure parameters for the model
##################################################
SEED:           int     = args.seed
torch.manual_seed(SEED)

CUDA_INDEX:         int     = args.cuda_index
DIMENSION:          int     = args.dimension
RESOLUTION:         int     = args.resolution
GAMMA:              float   = 0.0
DENSITY:            float   = args.density
OPERATOR:           str     = args.operator
BATCH_SIZE:         int     = args.batch_size
NUM_EPOCHS:         int     = args.num_epochs
LEARNING_RATE:      float   = args.learning_rate
SCHEDULER_STEP:     int     = args.scheduler_step
SCHEDULER_DECAY:    float   = args.scheduler_decay
PERIOD_REPORT:      int     = args.period_report
PERIOD_BACKUP:      int     = args.period_backup

NAME_BASE = f"FPL{DIMENSION}D__gamma{GAMMA:.1f}__res{RESOLUTION:03d}__seed{SEED}"

__PATH_ROOT = Path.cwd()
PATH_BASE = __PATH_ROOT / f"{DIMENSION}D__gamma{GAMMA:.1f}"
if not PATH_BASE.exists():
    raise FileNotFoundError(f"The specified gamma folder does not exist: [{str(PATH_BASE)}]")

path_train_set      = PATH_BASE / f"data__FPL{DIMENSION}D__gamma{GAMMA:.1f}__res{RESOLUTION:03d}__train.pth"
path_validation_set = PATH_BASE / f"data__FPL{DIMENSION}D__gamma{GAMMA:.1f}__res{RESOLUTION:03d}__validation.pth"
train_set           = Dataset_FPL(path_train_set, DIMENSION, RESOLUTION, DENSITY)
validation_set      = Dataset_FPL(path_validation_set, DIMENSION, RESOLUTION, DENSITY)
train_loader        = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
validation_loader   = DataLoader(validation_set, batch_size=BATCH_SIZE, shuffle=False)
device              = torch.device(f'cuda:{CUDA_INDEX}')

path_save = PATH_BASE / f"seed{SEED}"
path_save.mkdir(parents=True, exist_ok=True)
path_checkpoint = path_save / "checkpoint"
path_checkpoint.mkdir(parents=True, exist_ok=True)

sys.stdout = open(path_save/f"op_{OPERATOR}__{NAME_BASE}.log", 'w')

print(f"==========< Configuration >==========")
print(f"* Device           >>> {torch.cuda.get_device_name()} ({CUDA_INDEX})")
print(f"* Dimension        >>> {DIMENSION}")
print(f"* Resolution       >>> {RESOLUTION}")
print(f"* Gamma            >>> {GAMMA}")
print(f"* Operator         >>> {OPERATOR}")
print(f"* Batch size       >>> {BATCH_SIZE}")
print(f"* Number of epochs >>> {NUM_EPOCHS}")
print(f"* Learning rate    >>> {LEARNING_RATE:.2e}")
print(f"* Scheduler step   >>> {SCHEDULER_STEP}")
print(f"* Scheduler decay  >>> {SCHEDULER_DECAY}")
print(f"* Size of dataset  >>> train: {len(train_set)}, validation: {len(validation_set)}")
print(f"=====================================", flush=True)


##################################################
# Train the model
##################################################
if OPERATOR=='D':   op = generate_operator_D(DIMENSION, RESOLUTION, device=device)
elif OPERATOR=='F': op = generate_operator_F(DIMENSION, RESOLUTION, device=device)
else:               raise ValueError(f"Unknown operator type: {OPERATOR} (should be 'D' or 'F')")
optimizer = torch.optim.Adam(op.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=SCHEDULER_STEP, gamma=SCHEDULER_DECAY)

best_model:         torch.nn.Module | None  = None
best_error:         float | None            = None
best_epoch:         int | None              = None

log: dict[str, list[float]] = {
    'train_loss':       [],
    'train_error':      [],
    'validation_error': [],
    'train_time':       [],
}

total_train_time: float = 0.0
for epoch in range(1, 1+NUM_EPOCHS):
    # Train the model
    op.train()
    _train_time = time()
    _train_loss:    float = 0.0
    _train_error:   float = 0.0
    f: torch.Tensor
    for f, Df, Ff in train_loader:
        batch_size = f.size(0)
        target: torch.Tensor
        if OPERATOR=='D':   target = Df
        elif OPERATOR=='F': target = Ff
        else:               raise ValueError(f"Unknown operator type: {OPERATOR} (should be 'D' or 'F')")
        f = f.to(device);   target = target.to(device)
        op_f    = op.forward(f)
        loss    = mse_loss(op_f, target)
        error   = relative_error(op_f, target)
        optimizer.zero_grad()
        error.backward()
        optimizer.step()
        _train_loss  += loss.item() * batch_size
        _train_error += error.item() * batch_size
    _train_time = time() - _train_time
    _train_loss    /= len(train_set)
    _train_error   /= len(train_set)
    scheduler.step()
    total_train_time += _train_time
    
    # Evaluate the model
    op.eval()
    _validation_error:  float = 0.0
    with torch.inference_mode():
        f: torch.Tensor
        for f, Df, Ff in validation_loader:
            batch_size = f.size(0)
            f = f.to(device)
            target: torch.Tensor
            if OPERATOR=='D':   target = Df
            elif OPERATOR=='F': target = Ff
            else:               raise ValueError(f"Unknown operator type: {OPERATOR} (should be 'D' or 'F')")
            target = target.to(device)
            op_f    = op.forward(f)
            error   = relative_error(op_f, target)
            _validation_error += error.item() * batch_size
    _validation_error   /= len(validation_set)
    
    # Update logs
    log['train_loss'].append(_train_loss)
    log['train_error'].append(_train_error)
    log['validation_error'].append(_validation_error)
    log['train_time'].append(_train_time)
    
    # Update the best model
    if best_error is None or _validation_error<best_error:
        best_model = deepcopy(op)
        best_error = _validation_error
        best_epoch = epoch
    
    # Report the progress every `PERIOD_REPORT` epochs
    if epoch==1 or epoch%PERIOD_REPORT==0:
        print(
            f"[ Epoch {epoch:05d}/{NUM_EPOCHS:05d}]",
            f"train_loss: {_train_loss:.4e}, train_error: {_train_error:.4e}, validation_error: {_validation_error:.4e}",
            "",
            sep='\n', flush=True,
        )
    # Backup the model every `PERIOD_BACKUP` epochs
    if epoch==1 or epoch%PERIOD_BACKUP==0:
        torch.save(
            {
                'state_dict':   best_model.cpu().state_dict(),
                'best_epoch':   best_epoch,
                'log':          log,
            },
            path_checkpoint / f"op_{OPERATOR}__{NAME_BASE}__epoch{epoch:05d}.pth",
        )
        op.to(device)

op = best_model
op.cpu()
torch.save(
    {
        'state_dict':   op.state_dict(),
        'best_epoch':   best_epoch,
        'log':          log,
    },
    path_save / f"op_{OPERATOR}__{NAME_BASE}.pth",
)
print(f"Training completed. (Elapsed {sec_to_hms(int(total_train_time))})", flush=True)


##################################################
##################################################
# End of file