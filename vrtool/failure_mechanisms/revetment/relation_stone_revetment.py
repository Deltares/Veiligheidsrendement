from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RelationStoneRevetment:
    """Stores data for relation stone revetment"""

    slope_part: int
    year: int
    top_layer_thickness: float
    beta: float
