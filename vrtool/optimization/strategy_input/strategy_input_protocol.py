from typing import Protocol

from typing_extensions import runtime_checkable


@runtime_checkable
class StrategyInputProtocol(Protocol):
    design_method: str
