#from metrics.lossless_fourier_compression_metric import LosslessFourierCompression
#from numpy import load

#with open("data/rule1_ic8932_identity.npy", 'rb') as spacetime_file:
#    LosslessFourierCompression(spacetime_evolution=load(spacetime_file),verbose=True)

from pandas import DataFrame 
from json import load 

class SummariseEvaluations:
    def __init__(self, evaluations_path:str) -> None:
        self.evaluations_path = evaluations_path

    def symmetry_equivalence(self) -> DataFrame:
        with open(f"{self.evaluations_path}/symmetry_equivalence.json") as evaluation_file:
            evaluation = load(evaluation_file)
        return DataFrame.from_records(evaluation["averages"]).T

    def compression_ratio(self) -> DataFrame:
        with open(f"{self.evaluations_path}/highest_compression_ratio.json") as evaluation_file:
            evaluation = load(evaluation_file)
        summary = DataFrame([evaluation]).T
        summary.columns=["Highest Compression Ratio for N Rules"]
        return summary

    def sanity_check(self) -> DataFrame:
        with open(f"{self.evaluations_path}/sanity_check.json") as evaluation_file:
            evaluation = load(evaluation_file)
        summary = dict()
        for metric in evaluation:
            upper_limit = evaluation[metric]["limits"]["max_complexity"]
            lower_limit = evaluation[metric]["limits"]["min_complexity"]
            too_small = len(evaluation[metric]["too_small"])-1
            too_large = len(evaluation[metric]["too_large"])-1
            summary[metric] = dict(
                min_complexity=lower_limit,
                n_below_min_complexity = too_small,
                max_complexity=upper_limit,
                m_above_max_complexity = too_large,
            )
        return DataFrame.from_records(summary).T


    def pearsons_correlation(self) -> DataFrame:
        with open(f"{self.evaluations_path}/pearsons_correlations.json") as evaluation_file:
            evaluation = load(evaluation_file)
        summary = DataFrame([evaluation]).T
        summary.columns=["Pearsons Correlation"]
        return summary

