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
    cost: list[float]
    lcc: float

    def __init__(
        self,
        primary_measure: MeasureAsInputProtocol,
        secondary_measure: MeasureAsInputProtocol | None = None,
    ) -> None:
        self.primary = primary_measure
        self.secondary = secondary_measure
