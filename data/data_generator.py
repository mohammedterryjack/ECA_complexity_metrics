from numpy import fliplr, rot90, roll, save
from eca import OneDimensionalElementaryCellularAutomata 

def generate(path:str,width:int,depth:int,initial_condition:int) -> None:
    symmetry_transformations = dict(
        identity    =lambda image:image,
        inversion   =lambda image:1-image,
        rotation    =lambda image:rot90(image),
        reflection  =lambda image:fliplr(image),
        shift       =lambda image:roll(image,shift=10,axis=1),
    )
    for rule in range(256):
        ca = OneDimensionalElementaryCellularAutomata(
            initial_configuration=initial_condition,
            lattice_width=width
        )
        for _ in range(depth):
            ca.transition(rule)
        for symmetry,transform in symmetry_transformations.items():
            with open(f"{path}/rule{rule}_ic{initial_condition}_{symmetry}.npy", 'wb') as spacetime_file:
                save(spacetime_file, transform(image=ca.evolution()), allow_pickle=False)

generate(path="data",width=100,depth=100,initial_condition=8932)