from typing import Protocol
from typing_extensions import runtime_checkable


@runtime_checkable
class StrategyProtocol(Protocol):
    # In `StrategyBase` this is `type` but we want to avoid using "protected" names
    # for our own properties / attributes.
    design_method: str

    def evaluate(self, *args, **kwargs):
        """
        Evaluates the provided measures.
        TODO: For now the arguments are not specific as we do not have a clear view
        on which generic input structures will be used across the different strategies.
        """
        pass
