from pathlib import Path

from vrtool.decision_making.solutions import Solutions
from vrtool.decision_making.strategies.strategy_protocol import StrategyProtocol
from vrtool.run_workflows.vrtool_run_result_protocol import VrToolRunResultProtocol


class ResultsOptimization(VrToolRunResultProtocol):
    results_strategies: list[StrategyProtocol]
    results_solutions: dict[str, Solutions]

    def __init__(self) -> None:
        self.results_strategies = []

    @property
    def _step_output_filepath(self) -> Path:
        """
        Internal property to define where is located the output for the Optimization step.

        Returns:
            Path: Instance representing the file location.
        """
        return self.vr_config.output_directory / "FINAL_RESULT.out"
