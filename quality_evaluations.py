from typing import Dict
from pandas import DataFrame
from json import load, dump
from numpy import corrcoef 

class QualityEvaluations:
    def __init__(self, results_path:str) -> None:
        self.results_path = results_path

    def run_all(self) -> None:
        complexities = self.load_complexities(results_path=self.results_path)

        correlation_results = self.pearson_correlation(complexities=complexities)
        with open(f"{self.results_path}/pearsons_correlations.json","w") as results_file:
            dump(correlation_results, results_file, indent = 3)

        symmetry_results = self.symmetry_equivalence(complexities=complexities)
        with open(f"{self.results_path}/symmetry_equivalence.json","w") as results_file:
            dump(symmetry_results, results_file, indent = 3)

    @staticmethod
    def load_complexities(results_path) -> DataFrame:
        with open(f"{results_path}/complexities.json") as results_file:
            results = load(results_file)
        return DataFrame(results).T

    @staticmethod
    def pearson_correlation(complexities:DataFrame) -> Dict[str,float]:
        correlations = corrcoef(list(map(
            lambda metric_name:complexities[metric_name].tolist(),
            complexities
        )))
        return dict(zip(list(complexities),correlations[0]))

    @staticmethod
    def symmetry_equivalence(complexities:DataFrame) -> Dict[int,Dict[str,Dict[str,float]]]:
        results = dict()
        for rule_number in range(256):
            rule = f"rule{rule_number}_"
            results[rule] = dict()
            rule_complexities = complexities[complexities.index.str.contains(rule)]
            identity_rule_complexities = complexities[complexities.index.str.contains(rule) & complexities.index.str.contains("identity")]
            for metric_name in complexities:
                rule_complexity_differences = identity_rule_complexities[metric_name][0] - rule_complexities[metric_name]
                rule_complexity_differences = rule_complexity_differences.to_dict()
                rule_complexity_differences['total'] = sum(map(abs,rule_complexity_differences.values()))
                results[rule][metric_name] = rule_complexity_differences
        return results

    @staticmethod
    def sanity_check() -> None:
        pass #TODO

    @staticmethod
    def compression_ratio() -> None:
        pass #TODO


QualityEvaluations(results_path="results").run_all()