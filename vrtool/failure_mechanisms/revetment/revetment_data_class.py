from __future__ import annotations

import sys
from dataclasses import dataclass, field

from vrtool.failure_mechanisms.revetment.slope_part import SlopePart
from vrtool.failure_mechanisms.revetment.relation_grass_revetment import (
    RelationGrassRevetment,
)
from vrtool.failure_mechanisms.revetment.relation_stone_revetment import (
    RelationStoneRevetment,
)


@dataclass
class RevetmentDataClass:
    slope_parts: list[SlopePart] = field(default_factory=lambda: [])
    grass_relations: list[RelationGrassRevetment] = field(default_factory=lambda: [])
    stone_relations: list[RelationStoneRevetment] = field(default_factory=lambda: [])

    @property
    def current_transition_level(self) -> float:
        min_value_grass = sys.float_info.max
        for slope_part in self.slope_parts:
            if slope_part.is_grass:
                if slope_part.begin_part < min_value_grass:
                    min_value_grass = slope_part.begin_part

        if min_value_grass == sys.float_info.max:
            raise ValueError("No slope part with grass found")

        return min_value_grass
