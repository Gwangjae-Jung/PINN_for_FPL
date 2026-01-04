from    typing              import  Self
import  torch
from    torch.utils.data    import  Dataset


__all__: list[str] = ["mse_loss", "relative_error", "Dataset_FPL"]


def mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    assert pred.shape==target.shape, f"Shape mismatch: pred.shape={pred.shape}, target.shape={target.shape}"
    ndim = pred.ndim
    return torch.mean((pred-target).norm(p=2, dim=[i for i in range(1, ndim)]))


def relative_error(pred: torch.Tensor, target: torch.Tensor, order: float=2.0) -> torch.Tensor:
    assert pred.shape==target.shape, f"Shape mismatch: pred.shape={pred.shape}, target.shape={target.shape}"
    ndim = pred.ndim
    dim = tuple((i for i in range(1, ndim)))
    numerator = (pred-target).norm(p=order, dim=dim)
    denominator = target.norm(p=order, dim=dim)
    return torch.mean(numerator/denominator)


class Dataset_FPL(Dataset):
    def __init__(
            self,
            path_dataset:   str,
            dimension:      int,
            resolution:     int,
            density:        float,
        ) -> Self:
        raw_data = torch.load(path_dataset, weights_only=False)
        _inflate     = tuple([resolution for _ in range(dimension)])
        _permutation = tuple([0, 1+dimension, *(i+1 for i in range(dimension))])
        self.__f:   torch.Tensor = raw_data['f']
        self.__Df:  torch.Tensor = raw_data['Df']
        self.__Ff:  torch.Tensor = raw_data['Ff']
        size = self.__f.size(0)
        # NOTE: The dataset assumes the distribution functions have unit density
        self.__f    = density * self.__f.reshape(size, *_inflate, 1)
        self.__Df   = density * self.__Df.reshape(size, *_inflate, dimension**2)
        self.__Ff   = density * self.__Ff.reshape(size, *_inflate, dimension)
        self.__f    = self.__f.permute(_permutation)
        self.__Df   = self.__Df.permute(_permutation)
        self.__Ff   = self.__Ff.permute(_permutation)
        return
    

    def __len__(self) -> int:
        return self.__f.size(0)


    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.__f[idx], self.__Df[idx], self.__Ff[idx]


##################################################
##################################################
# End of file