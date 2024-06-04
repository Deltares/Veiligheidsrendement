from dataclasses import dataclass
from typing import Iterator

import pytest

from vrtool.common.enums.combinable_type_enum import CombinableTypeEnum
from vrtool.common.enums.measure_type_enum import MeasureTypeEnum
from vrtool.optimization.measures.measure_as_input_protocol import (
    MeasureAsInputProtocol,
)
from vrtool.optimization.measures.mechanism_per_year_probability_collection import (
    MechanismPerYearProbabilityCollection,
)
from vrtool.optimization.measures.sg_measure import SgMeasure
from vrtool.optimization.measures.sh_measure import ShMeasure


@dataclass
class OverridenShMeasure(ShMeasure):
    measure_type: MeasureTypeEnum
    measure_result_id: int = -1
    year: int = 0
    cost: float = float("nan")
    start_cost: float = 0
    combine_type: CombinableTypeEnum = CombinableTypeEnum.FULL
    discount_rate: float = 0
    mechanism_year_collection: MechanismPerYearProbabilityCollection = None
    beta_target: float = float("nan")
    transition_level: float = float("nan")
    dcrest: float = 0
    l_stab_screen: float = float("nan")


@dataclass
class OverridenSgMeasure(SgMeasure):
    measure_type: MeasureTypeEnum
    measure_result_id: int = -1
    year: int = 0
    cost: float = float("nan")
    start_cost: float = 0
    combine_type: CombinableTypeEnum = CombinableTypeEnum.FULL
    discount_rate: float = float("nan")
    mechanism_year_collection: MechanismPerYearProbabilityCollection = None
    dberm: float = float("nan")
    l_stab_screen: float = float("nan")


@pytest.fixture
def make_sh_measure() -> Iterator[type[MeasureAsInputProtocol]]:
    yield OverridenShMeasure


@pytest.fixture
def make_sg_measure() -> Iterator[type[MeasureAsInputProtocol]]:
    yield OverridenSgMeasure
