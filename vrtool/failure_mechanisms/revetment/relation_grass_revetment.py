from __future__ import annotations

from dataclasses import dataclass

from vrtool.failure_mechanisms.revetment.relation_revetment_protocol import (
    RelationRevetmentProtocol,
)


@dataclass
class RelationGrassRevetment(RelationRevetmentProtocol):
    """Stores data for relation grass revetment"""

    year: int
    transition_level: float
    beta: float
