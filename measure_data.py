from typing import Dict
from glob import glob 
from json import dump
from numpy import load 
from metrics.lossless_fourier_compression_metric import LosslessFourierCompression

def measure_dataset(data_path:str, complexity_metrics:Dict[str,callable]) -> None:
    data_results = dict()
    for filename in glob(f"{data_path}/*.npy"):
        with open(filename, 'rb') as spacetime_file:
            spacetime_evolution = load(spacetime_file)
        data_results[filename] = dict(map(
            lambda metric_name,complexity_metric:(
                metric_name,
                complexity_metric(spacetime_evolution)
            ),
            complexity_metrics.keys(),
            complexity_metrics.values()
        ))    
    with open(f"{data_path}/results.json","w") as results_file:
        dump(data_results, results_file, indent = 3)

measure_dataset(
    data_path="data",
    complexity_metrics = dict(
        foo =lambda s:0.0,
        bar =lambda s:1.0,
    )
)


#results = LosslessFourierCompression(spacetime_evolution=s, verbose=True)
#print(results.complexity) 

#SYMMETRY
#    name = filename.lstrip(f"{path}/")
#    rule = name.split("_")[0]
