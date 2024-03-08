from typing import Protocol

from typing_extensions import runtime_checkable


@runtime_checkable
class StrategyProtocol(Protocol):
    design_method: str

    def evaluate(self, *args, **kwargs):
        """
        Evaluates the provided measures.
        TODO: For now the arguments are not specific as we do not have a clear view
        on which generic input structures will be used across the different strategies.
        """
        pass

    def get_total_lcc_and_risk(self, step_number: int) -> tuple[float, float]:
        """
        _summary_
        THIS METHOD WILL BE DEPRECATED

        Args:
            step_number (int): _description_

        Returns:
            tuple[float, float]: _description_
        """
        pass
