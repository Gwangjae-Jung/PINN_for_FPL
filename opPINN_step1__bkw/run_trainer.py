import  argparse
import  subprocess

import  sys
sys.path.append('..')
from    config.op_config    import  *


parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=0, help='The random seed for reproducibility.')
parser.add_argument('--cuda_index', type=int, default=3, help='CUDA device index.')
parser.add_argument('--dimension', type=int, help='Dimension of the problem.')
parser.add_argument('--sample_v', type=int, default=64, help='The sampling resolution in the velocity space.')
parser.add_argument('--op', type=str, choices=['D', 'F'], help='The operator to be trained.')
parser.add_argument('--batch_size', type=int, default=OP__BATCH_SIZE, help='Batch size for training.')
args = parser.parse_args()


subprocess.run(
    [
        "python", "train_operators.py",
        
        "--seed",           str(args.seed),
        "--cuda_index",     str(args.cuda_index),
        
        "--dimension",      str(args.dimension),
        "--resolution",     str(args.sample_v),
        
        "--operator",       str(args.op),
        
        "--num_epochs",     str(OP__NUM_EPOCHS),
        "--learning_rate",  str(OP__LEARNING_RATE),
        "--batch_size",     str(OP__BATCH_SIZE),
        
        "--period_report",  str(OP__PERIOD_REPORT),
        "--period_backup",  str(OP__PERIOD_BACKUP),
    ]
)