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
    secondary: MeasureAsInputProtocol | None
    mechanism_year_collection: MechanismPerYearProbabilityCollection

    @property
    def lcc(self) -> float:
        if self.secondary is not None:
            return self.primary.lcc + self.secondary.lcc
        return self.primary.lcc

    @classmethod
    def from_input(
        cls,
        primary: MeasureAsInputProtocol,
        secondary: MeasureAsInputProtocol | None,
    ) -> CombinedMeasure:
        _mech_year_coll = primary.mechanism_year_collection
        if secondary is not None:
            _mech_year_coll = MechanismPerYearProbabilityCollection.combine(
                primary.mechanism_year_collection,
                secondary.mechanism_year_collection,
            )

        return cls(
            primary=primary,
            secondary=secondary,
            mechanism_year_collection=_mech_year_coll,
        )
