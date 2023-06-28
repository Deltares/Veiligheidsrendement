from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from vrtool.failure_mechanisms.revetment.relation_revetment_protocol import (
    RelationRevetmentProtocol,
)


@runtime_checkable
@dataclass
class SlopePartProtocol(Protocol):
    """stores data for slope part"""

    begin_part: float
    end_part: float
    tan_alpha: float
    top_layer_type: float  # note that e.g. 26.1 is a valid type
    top_layer_thickness: float = float("nan")

    slope_part_relations: list[RelationRevetmentProtocol] = field(
        default_factory=lambda: []
    )

    def is_valid(self) -> bool:
        """
        Validates whether this `SlopePartProtocol` instance fulfill its predefined requirements.

        Returns:
            bool: Wheter it is valid or not.
        """
        pass
