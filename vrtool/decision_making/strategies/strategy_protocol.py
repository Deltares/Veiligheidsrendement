from typing import Protocol

from typing_extensions import runtime_checkable

from vrtool.decision_making.strategies.strategy_step import StrategyStep
from vrtool.optimization.measures.section_as_input import SectionAsInput


@runtime_checkable
class StrategyProtocol(Protocol):
    design_method: str
    sections: list[SectionAsInput]
    time_periods: list[int]
    initial_step: StrategyStep
    optimization_steps: list[StrategyStep]

    def evaluate(self, *args, **kwargs):
        """
        Evaluates the provided measures.
        TODO: For now the arguments are not specific as we do not have a clear view
        on which generic input structures will be used across the different strategies.
        """
