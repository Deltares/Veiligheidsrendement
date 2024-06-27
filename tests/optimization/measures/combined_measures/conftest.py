from dataclasses import dataclass
from typing import Iterator

import pytest

from vrtool.common.enums.measure_type_enum import MeasureTypeEnum
from vrtool.optimization.measures.measure_as_input_protocol import (
    MeasureAsInputProtocol,
)
from vrtool.optimization.measures.mechanism_per_year_probability_collection import (
    MechanismPerYearProbabilityCollection,
)


@pytest.fixture(name="mocked_measure_as_input")
def _get_mocked_measure_as_input() -> Iterator[type[MeasureAsInputProtocol]]:
    @dataclass(kw_only=True)
    class MockMeasure(MeasureAsInputProtocol):
        measure_result_id: int = 0
        measure_type: MeasureTypeEnum = None
        cost: float = float("nan")
        base_cost: float = float("nan")
        discount_rate: float = float("nan")
        year: int = 0
        mechanism_year_collection: MechanismPerYearProbabilityCollection = None
        l_stab_screen: float = 0.0

    yield MockMeasure
