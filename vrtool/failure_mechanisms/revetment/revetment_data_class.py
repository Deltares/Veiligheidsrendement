from __future__ import annotations

import math
from dataclasses import dataclass, field

from vrtool.failure_mechanisms.revetment.relation_grass_revetment import (
    RelationGrassRevetment,
)
from vrtool.failure_mechanisms.revetment.slope_part import (
    GrassSlopePart,
    SlopePartProtocol,
    StoneSlopePart,
)


@dataclass
class RevetmentDataClass:
    slope_parts: list[SlopePartProtocol] = field(default_factory=lambda: [])
    grass_relations: list[RelationGrassRevetment] = field(default_factory=lambda: [])

    @property
    def current_transition_level(self) -> float:
        _grass_begin_parts = [
            _slope_part.begin_part
            for _slope_part in self.slope_parts
            if isinstance(_slope_part, GrassSlopePart)
        ]
        if not _grass_begin_parts:
            raise ValueError("No slope part with grass found.")

        return min(_grass_begin_parts)

    def get_transition_level_below_threshold(self, threshold: float) -> float:
        """
        Returns the greatest transition level value lower than the threshold.

        Args:
            threshold (float): Threshold value usually representing the crest height.

        Returns:
            float: greatest transition level.
        """
        _transition_levels = set(
            (
                gr.transition_level
                for gr in self.grass_relations
                if (gr.transition_level < threshold)
                or (math.isclose(gr.transition_level, threshold))
            )
        )
        if not _transition_levels:
            raise ValueError("No values found below the threshold {}".format(threshold))
        return max(_transition_levels)

    def get_available_years(self) -> list[int]:
        """
        Returns a list of the years whose data is available within its revetments (`RelationRevetmentProtocol`) for this `RevetmentDataClass` instance.

        Raises:
            ValueError: When the available years differ between revetments.

        Returns:
            list[int]: Available years with revetment data.
        """
        given_years_stone = set()
        for _slope_part in self.slope_parts:
            if isinstance(_slope_part, StoneSlopePart):
                for rel in _slope_part.slope_part_relations:
                    given_years_stone.add(rel.year)
                break
        given_years_grass = set()
        for rel in self.grass_relations:
            given_years_grass.add(rel.year)

        if given_years_grass == given_years_stone:
            return list(given_years_stone)

        raise ValueError("Years for grass and stone differ.")
