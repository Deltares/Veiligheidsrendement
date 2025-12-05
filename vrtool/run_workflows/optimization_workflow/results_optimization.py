from vrtool.decision_making.solutions import Solutions
from vrtool.decision_making.strategies.strategy_protocol import StrategyProtocol
from vrtool.run_workflows.vrtool_run_result_protocol import VrToolRunResultProtocol


class ResultsOptimization(VrToolRunResultProtocol):
    results_strategies: list[StrategyProtocol]
    results_solutions: dict[str, Solutions]

    def __init__(self) -> None:
        self.results_strategies = []
