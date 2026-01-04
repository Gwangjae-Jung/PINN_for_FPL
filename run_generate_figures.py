import  argparse
from    itertools   import  product
from    tqdm        import  tqdm
import  subprocess


LIST_GAMMA      = [-3.0+k for k in range(5)]
LIST_SAMPLE_T   = ['fixed_t', 'random_t']
LIST_RES_T      = [601]
LIST_RES_V      = [64]
LIST_INIT_TYPE  = ['bimaxwellian', 'bkw']


parser = argparse.ArgumentParser()
parser.add_argument('--cuda_index', type=int, default=3, help='The index of the GPU to use.')
args = parser.parse_args()
cuda_index: int     = args.cuda_index
for gamma, sample_t, res_t, res_v, init_type in tqdm(
        product(
            LIST_GAMMA,
            LIST_SAMPLE_T,
            LIST_RES_T,
            LIST_RES_V,
            LIST_INIT_TYPE
        )
    ):
    if init_type=='bkw' and gamma!=0.0: continue
    subprocess.run([
        "python", "generate_figures.py",
        "--gamma",      str(gamma),
        "--sample_t",   str(sample_t),
        "--res_t",      str(res_t),
        "--res_v",      str(res_v),
        "--init_type",  str(init_type),
    ])
print(f"Figure generation jobs completed.")