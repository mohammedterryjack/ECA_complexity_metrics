from pandas import DataFrame 
from json import load 
from numpy import load as np_load
from matplotlib.pyplot import imshow, show, title

class LatexTableFormatter:
    def __init__(self, evaluations_path:str) -> None:
        self.evaluations_path = evaluations_path

    def symmetry_equivalence(self) -> str:
        with open(f"{self.evaluations_path}/symmetry_equivalence.json") as evaluation_file:
            evaluation = load(evaluation_file)
        summary = DataFrame.from_records(evaluation["averages"]).T
        return summary.style.to_latex()

    def limits(self) -> str:
        with open(f"{self.evaluations_path}/within_limits.json") as evaluation_file:
            evaluation = load(evaluation_file)
        summary = dict()
        for metric in evaluation:
            upper_limit = evaluation[metric]["limits"]["max_complexity"]
            lower_limit = evaluation[metric]["limits"]["min_complexity"]
            summary[metric] = dict(
                minComplexity=lower_limit,
                maxComplexity=upper_limit,
            )
        summary = DataFrame.from_records(summary).T
        return summary.style.to_latex()

    def within_limits(self) -> str:
        with open(f"{self.evaluations_path}/within_limits.json") as evaluation_file:
            evaluation = load(evaluation_file)
        summary = dict()
        for metric in evaluation:
            too_small = len(evaluation[metric]["too_small"])-1
            too_large = len(evaluation[metric]["too_large"])-1
            summary[metric] = dict(
                nBelowMinComplexity = too_small,
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
    
    def view_samples(self) -> None:
        with open(f"{self.evaluations_path}/samples_per_complexity.json") as evaluation_file:
            evaluation = load(evaluation_file)
        for complexity_type, examples in evaluation.items():
            for filename,complexity_value in examples.items():
                with open(filename, 'rb') as spacetime_file:
                    spacetime_evolution=np_load(spacetime_file)  
                rule = filename.lstrip("data/").split("_")[0]              
                imshow(spacetime_evolution,cmap="gray")
                title(f"{complexity_type} {rule} ({round(complexity_value,4)})")
                show()
                    
formatter = LatexTableFormatter(evaluations_path="results")
print(formatter.pearsons_correlation())
print(formatter.limits())
print(formatter.within_limits())
print(formatter.symmetry_equivalence())
formatter.view_samples()