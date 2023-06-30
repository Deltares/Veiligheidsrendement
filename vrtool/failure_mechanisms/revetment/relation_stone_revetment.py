from __future__ import annotations

from dataclasses import dataclass

from vrtool.failure_mechanisms.revetment.relation_revetment_protocol import (
    RelationRevetmentProtocol,
)


@dataclass
class RelationStoneRevetment(RelationRevetmentProtocol):
    """Stores data for relation stone revetment"""

    year: int
    top_layer_thickness: float
    beta: float
