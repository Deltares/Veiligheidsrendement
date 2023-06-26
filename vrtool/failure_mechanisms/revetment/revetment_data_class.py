from __future__ import annotations

from dataclasses import dataclass, field
from vrtool.failure_mechanisms.revetment.grass_slope_part import GrassSlopePart

from vrtool.failure_mechanisms.revetment.relation_grass_revetment import (
    RelationGrassRevetment,
)

from vrtool.failure_mechanisms.revetment.slope_part import SlopePartProtocol


@dataclass
class RevetmentDataClass:
    slope_parts: list[SlopePartProtocol] = field(default_factory=lambda: [])
    grass_relations: list[RelationGrassRevetment] = field(default_factory=lambda: [])

    @property
    def current_transition_level(self) -> float:
        _grass_parts = [
            _slope_part
            for _slope_part in self.slope_parts
            if isinstance(_slope_part, GrassSlopePart)
        ]
        if not _grass_parts:
            raise ValueError("No slope part with grass found")

        return min(_grass_parts)
