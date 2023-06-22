from __future__ import annotations

from dataclasses import dataclass

GRASS_TYPE = 20.0


@dataclass
class SlopePart:
    """stores data for slope part"""

    begin_part: float
    end_part: float
    tan_alpha: float
    top_layer_type: float  # note that e.g. 26.1 is a valid type
    top_layer_thickness: float = float("nan")

    @property
    def is_grass(self) -> bool:
        return self.top_layer_type == GRASS_TYPE
