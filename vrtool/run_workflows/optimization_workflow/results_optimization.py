import logging
import shelve
from pathlib import Path
from typing import Dict, List

from vrtool.decision_making.solutions import Solutions
from vrtool.decision_making.strategies.strategy_base import StrategyBase
from vrtool.run_workflows.vrtool_run_result_protocol import VrToolRunResultProtocol


class ResultsOptimization(VrToolRunResultProtocol):
    results_strategies: List[StrategyBase]
    results_solutions: Dict[str, Solutions]

    def __init__(self) -> None:
        self.results_solutions = {}
        self.results_strategies = []
    
    @property
    def _step_output_filepath(self) -> Path:
        """
        Internal property to define where is located the output for the Optimization step.

        Returns:
            Path: Instance representing the file location.
        """
        return self.vr_config.output_directory / "FINAL_RESULT.out"

    def load_results(self):
        if self._step_output_filepath.exists():
            _shelf = shelve.open(str(self._step_output_filepath))
            self.selected_traject = _shelf["SelectedTraject"]
            self.results_solutions = _shelf["AllSolutions"]
            self.results_strategies = _shelf["AllStrategies"]
            _shelf.close()
            logging.info(
                "Loaded SelectedTraject, AllSolutions and AllStrategies from file"
            )

    def save_results(self):
        _shelf = shelve.open(str(self._step_output_filepath), "n")
        _shelf["SelectedTraject"] = self.selected_traject
        _shelf["AllSolutions"] = self.results_solutions
        _shelf["AllStrategies"] = self.results_strategies
        _shelf.close()
