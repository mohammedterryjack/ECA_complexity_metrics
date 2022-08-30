from pybdm import BDM
from numpy import ndarray 

class BlockDecompositionMethod:
    def __init__(self) -> None:
        self.model = BDM(ndim=2)
    
    def shannon_entropy(self, image:ndarray) -> str:
        return self.model.ent(image)