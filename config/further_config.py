from    typing  import  Self
from    pathlib import  Path


__all__: list[str] = ['FurtherConfig']


##################################################
##################################################
class FurtherConfig():
    """YAML-based configuration loader for inference experiments.

    ## Description
    Reads a YAML configuration file and exposes its entries as typed read-only
    properties. Designed to complement `TrainParser` by providing additional
    hyperparameters (VHS model, surrogate operator paths) that are fixed for a given
    experimental setting rather than passed as command-line arguments.

    ## Arguments
    `path` (`str | Path`): Path to the YAML configuration file.

    ## Returns
    `None`: None.
    """
    def __init__(self, path: str | Path) -> Self:
        import  yaml
        further_config = yaml.safe_load(open(path, 'r'))

        self.__cuda_index:   int   = int(further_config['CUDA_INDEX'])
        self.__vhs_coeff:    float = float(further_config['VHS_COEFF'])
        self.__vhs_exponent: float = float(further_config['VHS_EXPONENT'])
        self.__init_type:    str   = str(further_config['INIT_TYPE'])
        self.__path_D:       str   = str(further_config['PATH_D'])
        self.__path_F:       str   = str(further_config['PATH_F'])
        return

    @property
    def cuda_index(self) -> int:    return self.__cuda_index
    @property
    def vhs_coeff(self) -> float:   return self.__vhs_coeff
    @property
    def vhs_exponent(self) -> float: return self.__vhs_exponent
    @property
    def init_type(self) -> str:     return self.__init_type
    @property
    def path_D(self) -> str:        return self.__path_D
    @property
    def path_F(self) -> str:        return self.__path_F


##################################################
##################################################
# End of file