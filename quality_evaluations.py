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

        anomalous_results = self.analyse_anomalous_complexities(complexities=complexities)
        with open(f"{self.results_path}/anomalous_complexities.json","w") as results_file:
            dump(anomalous_results, results_file, indent = 3)

        symmetry_results = self.symmetry_equivalence(complexities=complexities)
        with open(f"{self.results_path}/symmetry_equivalence.json","w") as results_file:
            dump(symmetry_results, results_file, indent = 3)

        extrema_results = self.within_limits(complexities=complexities)
        with open(f"{self.results_path}/within_limits.json","w") as results_file:
            dump(extrema_results, results_file, indent = 3)

    @staticmethod
    def load_complexities(results_path) -> DataFrame:
        with open(f"{results_path}/complexities.json") as results_file:
            results = load(results_file)
        return DataFrame(results).T

    @staticmethod 
    def analyse_anomalous_complexities(complexities:DataFrame) -> Dict[str, Dict[str,float]]:
        anomalous_compexities = dict(rule=dict(metrica=0.3,metricb=0.4,metricc=0.4))
        #TODO: normalise all complexity values
        #TODO: compare the complexities for each sample
        #TODO: get the difference from the proposed metric to others
        #TODO: when all metrics have a high difference <- add the result to differences
        #TODO: return differences
        return anomalous_compexities 

    @staticmethod
    def pearson_correlation(complexities:DataFrame) -> Dict[str,float]:
        correlations = corrcoef(list(map(
            lambda metric_name:complexities[metric_name].tolist(),
            complexities
        )))
        return dict(zip(list(complexities),correlations[0]))

    @staticmethod
    def symmetry_equivalence(complexities:DataFrame) -> Dict[int,Dict[str,Dict[str,float]]]:
        SYMMETRIES = ("identity","inversion","shift","rotation","reflection","total")
        results = dict()
        totals = dict()
        for rule_number in range(256):
            rule = f"rule{rule_number}_"
            results[rule] = dict()
            rule_complexities = complexities[complexities.index.str.contains(rule)]
            identity_rule_complexities = complexities[complexities.index.str.contains(rule) & complexities.index.str.contains("identity")]
            for metric_name in complexities:
                rule_complexity_differences = identity_rule_complexities[metric_name][0] - rule_complexities[metric_name]
                rule_complexity_differences = rule_complexity_differences.to_dict()
                total = sum(map(abs,rule_complexity_differences.values()))
                if metric_name not in totals:
                    totals[metric_name] = dict(map(lambda symmetry_name:(symmetry_name,0),SYMMETRIES))
                for key,value in rule_complexity_differences.items():
                    for symmetry_name in SYMMETRIES:
                        if symmetry_name in key:
                            totals[metric_name][symmetry_name] += abs(value)
                totals[metric_name]["total"] += total
                rule_complexity_differences["total"] = total
                results[rule][metric_name] = rule_complexity_differences        
        n_rules = len(results)
        for metric in totals:
            for total in totals[metric]:
                totals[metric][total] /= n_rules
        results['averages'] = totals
        return results

    @staticmethod
    def within_limits(complexities:DataFrame) -> Dict[str,Dict[str,Dict[str,float]]]:
        results = dict()
        for metric_name in complexities:
            metric_complexities = complexities[metric_name]
            min_complexity = metric_complexities[metric_complexities.index.str.contains("minimum_complexity")].tolist()[0]
            max_complexity = metric_complexities[metric_complexities.index.str.contains("maximum_complexity")].tolist()[0]
            results[metric_name] = dict(
                limits = dict(min_complexity=min_complexity,max_complexity=max_complexity),
                too_small=metric_complexities[metric_complexities<=min_complexity].to_dict(),
                too_large=metric_complexities[metric_complexities>=max_complexity].to_dict()
            )
        return results


QualityEvaluations(results_path="results").run_all()
