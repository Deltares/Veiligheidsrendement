import logging
import shelve
from typing import Dict, List

import numpy as np
import pandas as pd

from src.DecisionMaking.Solutions import Solutions
from src.DecisionMaking.Strategy import (
    GreedyStrategy,
    Strategy,
    TargetReliabilityStrategy,
)
from src.run_workflows.run_measures import ResultsMeasures
from src.run_workflows.vrtool_run_protocol import (
    VrToolPlotMode,
    VrToolRunProtocol,
    VrToolRunResultProtocol,
)


class ResultsOptimization(VrToolRunResultProtocol):
    results_strategies: List[Strategy]
    results_solutions: Dict[str, Solutions]

    def load_results(self):
        _step_3_results = self.vr_config.output_directory / "FINAL_RESULT.out"
        if _step_3_results.exists():
            _shelf = shelve.open(str(_step_3_results))
            self.selected_traject = _shelf["SelectedTraject"]
            self.results_solutions = _shelf["AllSolutions"]
            self.results_strategies = _shelf["AllStrategies"]
            _shelf.close()
            logging.info(
                "Loaded SelectedTraject, AllSolutions and AllStrategies from file"
            )

    def save_results(self):
        _step_3_results = self.vr_config.output_directory / "FINAL_RESULT.out"
        _shelf = shelve.open(str(_step_3_results), "n")
        _shelf["SelectedTraject"] = self.selected_traject
        _shelf["AllSolutions"] = self.results_solutions
        _shelf["AllStrategies"] = self.results_strategies
        _shelf.close()


class RunOptimization(VrToolRunProtocol):
    def __init__(
        self, results_measures: ResultsMeasures, plot_mode: VrToolPlotMode
    ) -> None:
        self._solutions_dict = results_measures.solutions_dict
        self.selected_traject = results_measures.selected_traject
        self.vr_config = results_measures.vr_config
        self._plot_mode = plot_mode

    def run(self) -> ResultsOptimization:
        _results_optimization = ResultsOptimization()
        if self.vr_config.reuse_output:
            _results_optimization.load_results()
        else:
            ## STEP 3: EVALUATE THE STRATEGIES
            for i in self.vr_config.design_methods:
                if i in [
                    "TC",
                    "Total Cost",
                    "Optimized",
                    "Greedy",
                    "Veiligheidsrendement",
                ]:
                    # Initialize a GreedyStrategy:
                    _greedy_optimization = GreedyStrategy(i)

                    # Combine available measures
                    _greedy_optimization.combine(
                        self.selected_traject,
                        self._solutions_dict,
                        filtering="off",
                        splitparams=True,
                    )

                    # Calculate optimal strategy using Traject & Measures objects as input (and possibly general settings)
                    _greedy_optimization.evaluate(
                        self.selected_traject,
                        self._solutions_dict,
                        splitparams=True,
                        setting="cautious",
                        f_cautious=1.5,
                        max_count=600,
                        BCstop=0.1,
                    )

                    # plot beta time for all measure steps for each strategy
                    if self._plot_mode == VrToolPlotMode.EXTENSIVE:
                        _greedy_optimization.plotBetaTime(
                            self.selected_traject,
                            typ="single",
                            path=self.vr_config.directory,
                        )

                    _greedy_optimization = self._replace_names(
                        _greedy_optimization, self._solutions_dict
                    )
                    cost_Greedy = _greedy_optimization.determineRiskCostCurve(
                        self.selected_traject
                    )

                    # write to csv's
                    _results_dir = self.vr_config.directory / "results"
                    _greedy_optimization.TakenMeasures.to_csv(
                        _results_dir.joinpath(
                            "TakenMeasures_" + _greedy_optimization.type + ".csv"
                        )
                    )
                    pd.DataFrame(
                        np.array(
                            [
                                cost_Greedy["LCC"],
                                cost_Greedy["TR"],
                                np.add(cost_Greedy["LCC"], cost_Greedy["TR"]),
                            ]
                        ).T,
                        columns=["LCC", "TR", "TC"],
                    ).to_csv(
                        _results_dir / "TotalCostValues_Greedy.csv",
                        float_format="%.1f",
                    )
                    _greedy_optimization.makeSolution(
                        _results_dir.joinpath(
                            "TakenMeasures_Optimal_"
                            + _greedy_optimization.type
                            + ".csv",
                        ),
                        step=cost_Greedy["TC_min"] + 1,
                        type="Optimal",
                    )
                    _greedy_optimization.makeSolution(
                        _results_dir.joinpath(
                            "FinalMeasures_" + _greedy_optimization.type + ".csv"
                        ),
                        type="Final",
                    )
                    for j in _greedy_optimization.options:
                        _greedy_optimization.options[j].to_csv(
                            _results_dir.joinpath(
                                j + "_Options_" + _greedy_optimization.type + ".csv",
                            )
                        )
                    costs = _greedy_optimization.determineRiskCostCurve(
                        self.selected_traject
                    )
                    TR = costs["TR"]
                    LCC = costs["LCC"]
                    pd.DataFrame(
                        np.array([TR, LCC]).reshape((len(TR), 2)), columns=["TR", "LCC"]
                    ).to_csv(_results_dir / "TotalRiskCost.csv")
                    _results_optimization.results_strategies.append(
                        _greedy_optimization
                    )

                elif i in ["OI", "TargetReliability", "Doorsnede-eisen"]:
                    # Initialize a strategy type (i.e combination of objective & constraints)
                    _target_reliability_based = TargetReliabilityStrategy(i)
                    # Combine available measures
                    _target_reliability_based.combine(
                        self.selected_traject,
                        self._solutions_dict,
                        filtering="off",
                        splitparams=True,
                    )

                    # Calculate optimal strategy using Traject & Measures objects as input (and possibly general settings)
                    _target_reliability_based.evaluate(
                        self.selected_traject, self._solutions_dict, splitparams=True
                    )
                    _target_reliability_based.makeSolution(
                        _results_dir.joinpath(
                            "FinalMeasures_" + _target_reliability_based.type + ".csv",
                        ),
                        type="Final",
                    )

                    # plot beta time for all measure steps for each strategy
                    if self._plot_mode == VrToolPlotMode.EXTENSIVE:
                        _target_reliability_based.plotBetaTime(
                            self.selected_traject,
                            typ="single",
                            path=self.vr_config.directory,
                        )

                    _target_reliability_based = self._replace_names(
                        _target_reliability_based, self._solutions_dict
                    )
                    # write to csv's
                    _target_reliability_based.TakenMeasures.to_csv(
                        _results_dir.joinpath(
                            "TakenMeasures_" + _target_reliability_based.type + ".csv",
                        )
                    )
                    for j in _target_reliability_based.options:
                        _target_reliability_based.options[j].to_csv(
                            _results_dir.joinpath(
                                j
                                + "_Options_"
                                + _target_reliability_based.type
                                + ".csv",
                            )
                        )

                    _results_optimization.results_strategies.append(
                        _target_reliability_based
                    )

        _results_optimization.selected_traject = self.selected_traject
        _results_optimization.vr_config = self.vr_config
        _results_optimization.results_solutions = self._solutions_dict
        if self.vr_config.shelves:
            _results_optimization.save_results()
        return _results_optimization

    def _replace_names(
        self, strategy_case: Strategy, solution_case: Solutions
    ) -> Strategy:
        strategy_case.TakenMeasures = strategy_case.TakenMeasures.reset_index(drop=True)
        for i in range(1, len(strategy_case.TakenMeasures)):
            _measure_id = strategy_case.TakenMeasures.iloc[i]["ID"]
            if isinstance(_measure_id, list):
                _measure_id = "+".join(_measure_id)

            section = strategy_case.TakenMeasures.iloc[i]["Section"]
            name = (
                solution_case[section]
                .MeasureTable.loc[
                    solution_case[section].MeasureTable["ID"] == _measure_id
                ]["Name"]
                .values
            )
            strategy_case.TakenMeasures.at[i, "name"] = name
        return strategy_case
