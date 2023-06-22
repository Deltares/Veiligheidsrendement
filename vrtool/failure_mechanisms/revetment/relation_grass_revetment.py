from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RelationGrassRevetment:
    """Stores data for relation grass revetment"""

    year: int
    transition_level: float
    beta: float
