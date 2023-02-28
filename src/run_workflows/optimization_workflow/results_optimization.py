import logging
import shelve
from typing import Dict, List

from src.DecisionMaking.Solutions import Solutions
from src.DecisionMaking.Strategy import Strategy
from src.run_workflows.vrtool_run_result_protocol import VrToolRunResultProtocol


class ResultsOptimization(VrToolRunResultProtocol):
    results_strategies: List[Strategy]
    results_solutions: Dict[str, Solutions]

    def __init__(self) -> None:
        self.results_solutions = {}
        self.results_strategies = []

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
