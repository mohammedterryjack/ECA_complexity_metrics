from typing import Dict
from glob import glob 
from json import dump
from numpy import load 
from metrics.lossless_fourier_compression_metric import LosslessFourierCompression
from metrics.run_length_encoding import RunLengthEncoding

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
    with open("results/complexities.json","w") as results_file:
        dump(data_results, results_file, indent = 3)

measure_dataset(
    data_path="data",
    complexity_metrics = dict(
        #LosslessFourierCompression =lambda s:LosslessFourierCompression(spacetime_evolution=s).complexity,
        Test=lambda s:0.3,
        RunLengthEncoding=lambda s:1/RunLengthEncoding.compression_ratio(image=s),
  )
)