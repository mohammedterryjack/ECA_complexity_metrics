from typing import List

R = List[List[float]]
S = List[List[int]]
C = List[List[complex]]
class 𝝋:
    def __init__(self, mask:S) -> None:
        self.m = mask
    
    def __hash__(self) -> str:
        return hash(str(self))

    def __eq__(self, other:"𝝋") -> bool:
        return str(self) == str(other)
    
    def __repr__(self) -> str:
        return str(self.m)
    
    def __call__(self, z:Z) -> Z:
        return z*self.m
