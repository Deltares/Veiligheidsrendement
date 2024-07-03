from dataclasses import dataclass
from typing import Callable, Iterator

import pytest

from vrtool.common.enums.measure_type_enum import MeasureTypeEnum
from vrtool.common.enums.mechanism_enum import MechanismEnum
from vrtool.optimization.measures.measure_as_input_protocol import (
    MeasureAsInputProtocol,
)
from vrtool.optimization.measures.mechanism_per_year_probability_collection import (
    MechanismPerYearProbabilityCollection,
)
from vrtool.optimization.measures.sg_measure import SgMeasure
from vrtool.optimization.measures.sh_measure import ShMeasure
from vrtool.optimization.measures.sh_sg_measure import ShSgMeasure


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


@pytest.fixture(name="sh_measure_factory")
def get_sh_measure_factory(
    probability_collection_factory: Callable[
        [MechanismEnum], MechanismPerYearProbabilityCollection
    ],
) -> Iterator[Callable[[], ShMeasure]]:
    """
    Fixture to generate `ShMeasure` instances with dummy values.
    """

    def create_sh_measure() -> ShMeasure:
        return ShMeasure(
            beta_target=float("nan"),
            transition_level=float("nan"),
            dcrest=float("nan"),
            cost=0,
            base_cost=0,
            measure_result_id=0,
            measure_type=None,
            combine_type=None,
            discount_rate=0.3,
            mechanism_year_collection=probability_collection_factory(
                MechanismEnum.OVERFLOW
            ),
            l_stab_screen=float("nan"),
            year=0,
        )

    yield create_sh_measure


@pytest.fixture(name="sg_measure_factory")
def get_sg_measure_factory(
    probability_collection_factory: Callable[
        [MechanismEnum], MechanismPerYearProbabilityCollection
    ],
) -> Iterator[Callable[[], SgMeasure]]:
    """
    Fixture to generate `SgMeasure` instances with dummy values.
    """

    def create_sg_measure() -> SgMeasure:
        return SgMeasure(
            dberm=float("nan"),
            cost=0,
            base_cost=0,
            measure_result_id=0,
            measure_type=None,
            combine_type=None,
            discount_rate=0.3,
            mechanism_year_collection=probability_collection_factory(
                MechanismEnum.OVERFLOW
            ),
            l_stab_screen=float("nan"),
            year=0,
        )

    yield create_sg_measure


@pytest.fixture(name="shsg_measure_factory")
def get_shsg_measure_factory(
    probability_collection_factory: Callable[
        [MechanismEnum], MechanismPerYearProbabilityCollection
    ],
) -> Iterator[Callable[[], ShSgMeasure]]:
    """
    Fixture to generate `ShSgMeasure` instances with dummy values.
    """

    def create_shsg_measure() -> ShSgMeasure:
        return ShSgMeasure(
            cost=0,
            base_cost=0,
            discount_rate=0,
            measure_result_id=0,
            measure_type=None,
            combine_type=None,
            mechanism_year_collection=probability_collection_factory(
                MechanismEnum.OVERFLOW
            ),
            l_stab_screen=float("nan"),
            year=0,
            dcrest=float("nan"),
            dberm=float("nan"),
        )

    yield create_shsg_measure
