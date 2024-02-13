from __future__ import annotations

from dataclasses import dataclass

from vrtool.optimization.measures.measure_as_input_protocol import (
    MeasureAsInputProtocol,
)
from vrtool.optimization.measures.mechanism_per_year_probability_collection import (
    MechanismPerYearProbabilityCollection,
)


@dataclass
class CombinedMeasure:
    primary: MeasureAsInputProtocol
    secondary: MeasureAsInputProtocol
    mechanism_year_collection: MechanismPerYearProbabilityCollection

    @property
    def lcc(self) -> float:
        return self.primary.lcc + self.secondary.lcc

    @classmethod
    def from_section_as_input(
        cls,
        primary: MeasureAsInputProtocol,
        secondary: MeasureAsInputProtocol,
    ) -> CombinedMeasure:
        return cls(
            primary=primary,
            secondary=secondary,
            mechanism_year_collection=MechanismPerYearProbabilityCollection.combine(
                primary.mechanism_year_collection,
                secondary.mechanism_year_collection,
            ),
        )
