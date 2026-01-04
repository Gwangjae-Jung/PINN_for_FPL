from    typing      import  Self
from    pathlib     import  Path


__all__: list[str] = ['FurtherConfig']


class FurtherConfig():
    def __init__(self, path: str | Path) -> Self:
        import  yaml
        further_config = yaml.load(open(path, 'r'), Loader=yaml.FullLoader)
        
        self.__cuda_index   = int(further_config['CUDA_INDEX'])
        self.__vhs_coeff    = float(further_config['VHS_COEFF'])
        self.__vhs_exponent = float(further_config['VHS_EXPONENT'])
        self.__init_type    = str(further_config['INIT_TYPE'])
        self.__path_D       = str(further_config['PATH_D'])
        self.__path_F       = str(further_config['PATH_F'])
        
        return
    
    
    @property
    def cuda_index(self) -> int: return self.__cuda_index
    @property
    def vhs_coeff(self) -> float: return self.__vhs_coeff
    @property
    def vhs_exponent(self) -> float: return self.__vhs_exponent
    @property
    def init_type(self) -> str:  return self.__init_type
    @property
    def is_exact(self) -> bool:  return self.__is_exact
    @property
    def path_D(self) -> str:     return self.__path_D
    @property
    def path_F(self) -> str:     return self.__path_F


##################################################
##################################################
# End of file