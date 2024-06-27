from typing import Callable, Iterable, Iterator

import pytest

from vrtool.common.enums.mechanism_enum import MechanismEnum
from vrtool.optimization.measures.combined_measures.combined_measure_base import (
    CombinedMeasureBase,
)
from vrtool.optimization.measures.combined_measures.sg_combined_measure import (
    SgCombinedMeasure,
)
from vrtool.optimization.measures.combined_measures.sh_combined_measure import (
    ShCombinedMeasure,
)
from vrtool.optimization.measures.measure_as_input_base import MeasureAsInputBase
from vrtool.optimization.measures.measure_as_input_protocol import (
    MeasureAsInputProtocol,
)
from vrtool.optimization.measures.mechanism_per_year import MechanismPerYear
from vrtool.optimization.measures.mechanism_per_year_probability_collection import (
    MechanismPerYearProbabilityCollection,
)
from vrtool.optimization.measures.sg_measure import SgMeasure
from vrtool.optimization.measures.sh_measure import ShMeasure


@pytest.fixture(name="probability_collection_factory")
def get_valid_probability_collection_factory() -> Iterator[
    Callable[[MechanismEnum], MechanismPerYearProbabilityCollection]
]:
    def create_mpy_prob_collection(
        mechanism_type: MechanismEnum,
    ) -> MechanismPerYearProbabilityCollection:
        _mech_per_year = MechanismPerYear(
            mechanism=mechanism_type, year=0, probability=0.5
        )
        return MechanismPerYearProbabilityCollection(probabilities=[_mech_per_year])

    yield create_mpy_prob_collection


@pytest.fixture(name="measure_as_input_factory")
def get_measure_as_input_factory_fixture(
    probability_collection_factory: Callable[
        [MechanismEnum], MechanismPerYearProbabilityCollection
    ]
) -> Iterable[Callable[[dict], MeasureAsInputProtocol]]:
    def create_measure_as_input(**kwargs) -> MeasureAsInputProtocol:
        default_values = dict(
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
        return MeasureAsInputBase(**(default_values | kwargs))

    yield create_measure_as_input


@pytest.fixture(name="combined_measure_factory")
def get_combined_measure_factory_fixture(
    measure_as_input_factory: Callable[[dict], MeasureAsInputProtocol]
) -> Iterable[Callable[[type[CombinedMeasureBase], dict, dict], CombinedMeasureBase]]:
    def create_combined_measure(
        combined_measure_type: type[ShCombinedMeasure | SgCombinedMeasure],
        primary_dict: dict,
        secondary_dict: dict,
    ) -> CombinedMeasureBase:
        # This method only supports `ShCombinedMeasure` and `SgCombinedMeasure`.
        return combined_measure_type(
            primary=measure_as_input_factory(**primary_dict),
            secondary=measure_as_input_factory(**secondary_dict)
            if secondary_dict
            else None,
            mechanism_year_collection=None,
            sequence_nr=-1,
        )

    yield create_combined_measure


@pytest.fixture(name="sh_measure_factory")
def get_sh_measure_factory(
    probability_collection_factory,
) -> Iterator[Callable[[], ShMeasure]]:
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
    probability_collection_factory,
) -> Iterator[Callable[[], SgMeasure]]:
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
