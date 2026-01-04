from    torch       import  pi


__all__:    list[str] = [    
    'DIMENSION',
    'INIT_TYPE',
    'MAX_T__DICT',
    'MAX_V__DICT',
    'SAMPLE_T',
    'SAMPLE_V',
    'SAMPLE_V_INIT',
    'DENSITY__DICT',
    'VHS_COEFF',
    'VHS_EXPONENT',
    
    'DEPTH',
    'WIDTH',
    'SOFTPLUS',
    'PATH_D',
    'PATH_F',
    
    'LEARNING_RATE',
    'NUM_EPOCHS',
    'NUM_ITERATIONS',

    'INIT_COND__DEV',
    'INIT_COND__STD',
    'BKW_COEFF_EXT',
]


# `group_equation`
DIMENSION:      int     = 2
INIT_TYPE:      str     = None
MAX_T__DICT:    dict[str, float] = {
    'maxwellian':   3.0,
    'bimaxwellian': 3.0,
    'bkw':          3.0,
}
MAX_V__DICT:    dict[str, float] = {
    'maxwellian':   5.0,
    'bimaxwellian': 5.0,
    'bkw':          2*pi,
}
SAMPLE_T:       int     = 10
SAMPLE_V:       int     = 64
SAMPLE_V_INIT:  int     = 1000
DENSITY__DICT:  dict[str, float] = {
    'maxwellian':   0.2,
    'bimaxwellian': 0.2,
    'bkw':          0.2,
}
VHS_COEFF:      float   = None
VHS_EXPONENT:   float   = None
## TODO: Set `VHS_COEFF`, `VHS_EXPONENT`, and `INIT_TYPE`


# `group_pinn`
DEPTH:      int     = 4
WIDTH:      int     = 100
SOFTPLUS:   float   = 1.0
PATH_D:     str     = None
PATH_F:     str     = None
## TODO: Set `IS_EXACT`, `PATH_D`, and `PATH_F`


# `group_train`
LEARNING_RATE:  float   = 1e-3
NUM_EPOCHS:     int     = int(5e3)
NUM_ITERATIONS: int     = 20


# `group_ic`
INIT_COND__DEV:     float   = 1.0
INIT_COND__STD:     float   = 0.8
BKW_COEFF_EXT:      float   = 0.5