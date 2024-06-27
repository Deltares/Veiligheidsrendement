from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from vrtool.optimization.measures.measure_as_input_protocol import (
    MeasureAsInputProtocol,
)
from vrtool.optimization.measures.mechanism_per_year_probability_collection import (
    MechanismPerYearProbabilityCollection,
)


@dataclass(kw_only=True)
class CombinedMeasureProtocol(Protocol):
    """
    Represents the combination between two supporting measures
    where the most important (`primary`) will have the leading
    base costs over the auxiliar (`secondary`) one.
    """

    primary: MeasureAsInputProtocol
    mechanism_year_collection: MechanismPerYearProbabilityCollection
    # Legacy index for mapping back to the old structure for evaluate
    sequence_nr: int = None

    @property
    def lcc(self) -> float:
        """
        Calculates the "life cycle cost" of this combined measure
        based on their respectives `cost` values and discount rate / year.

        Returns:
            float: The total (bruto) lcc of this combined measure.
        """

    def is_base_measure(self) -> bool:
        """
        Determines whether this `CombinedMeasure` could be considered
        as a base measure (usually when `dberm` / `dcrest` equal to 0).

        Returns:
            bool: True when its primary measure is an initial measure.
        """

    def compares_to(self, other: "CombinedMeasureProtocol") -> bool:
        """
        Compares this instance of a 'CombinedMeasureProtocol' with another one.
        This method could be also implemented as an overriding of the
        'equal' operator ('__eq__').

        Args:
            other (CombinedMeasureProtocol): Other combined measure to compare.

        Returns:
            bool: Whether both combined measures can be conisered as matching.
        """
