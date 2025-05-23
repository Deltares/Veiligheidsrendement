from typing import Protocol, runtime_checkable


@runtime_checkable
class RelationRevetmentProtocol(Protocol):
    year: int
    beta: float
