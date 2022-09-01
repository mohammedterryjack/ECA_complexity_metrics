from audioop import reverse
from pandas import DataFrame 
from json import load 

class LatexTableFormatter:
    def __init__(self, evaluations_path:str) -> None:
        self.evaluations_path = evaluations_path

    def symmetry_equivalence(self) -> str:
        with open(f"{self.evaluations_path}/symmetry_equivalence.json") as evaluation_file:
            evaluation = load(evaluation_file)
        summary = DataFrame.from_records(evaluation["averages"]).T
        return summary.style.to_latex()

    def sanity_check(self) -> str:
        with open(f"{self.evaluations_path}/within_limits.json") as evaluation_file:
            evaluation = load(evaluation_file)
        summary = dict()
        for metric in evaluation:
            upper_limit = evaluation[metric]["limits"]["max_complexity"]
            lower_limit = evaluation[metric]["limits"]["min_complexity"]
            too_small = len(evaluation[metric]["too_small"])-1
            too_large = len(evaluation[metric]["too_large"])-1
            summary[metric] = dict(
                minComplexity=lower_limit,
                nBelowMinComplexity = too_small,
                maxComplexity=upper_limit,
                mAboveMaxComplexity = too_large,
            )
        summary = DataFrame.from_records(summary).T
        return summary.style.to_latex()


    def pearsons_correlation(self) -> str:
        with open(f"{self.evaluations_path}/pearsons_correlations.json") as evaluation_file:
            evaluation = load(evaluation_file)
        summary = DataFrame([evaluation]).T
        summary.columns=["Pearsons Correlation"]
        return summary.sort_values(by="Pearsons Correlation",ascending=False).style.to_latex()
    

formatter = LatexTableFormatter(evaluations_path="results")
print(formatter.pearsons_correlation())
print(formatter.sanity_check())
print(formatter.symmetry_equivalence())
