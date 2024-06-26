from typing import Callable, Iterable

import pytest

from vrtool.optimization.measures.combined_measure import CombinedMeasure
from vrtool.optimization.measures.measure_as_input_base import MeasureAsInputBase
from vrtool.optimization.measures.measure_as_input_protocol import (
    MeasureAsInputProtocol,
)


@pytest.fixture(name="measure_as_input_factory")
def get_measure_as_input_factory_fixture() -> Iterable[
    Callable[[dict], MeasureAsInputProtocol]
]:
    def create_measure_as_input(**kwargs) -> MeasureAsInputProtocol:
        default_values = dict(
            measure_result_id=0,
            measure_type=None,
            combine_type=None,
            discount_rate=0.3,
            mechanism_year_collection=None,
            l_stab_screen=float("nan"),
        )
        return MeasureAsInputBase(**(default_values | kwargs))

    yield create_measure_as_input


@pytest.fixture(name="combined_measure_factory")
def get_combined_measure_factory_fixture(
    measure_as_input_factory: Callable[[dict, dict], MeasureAsInputProtocol]
) -> Iterable[Callable[[], CombinedMeasure]]:
    def create_combined_measure(
        primary_dict: dict, secondary_dict: dict
    ) -> CombinedMeasure:
        return CombinedMeasure(
            primary=measure_as_input_factory(**primary_dict),
            secondary=measure_as_input_factory(**secondary_dict)
            if secondary_dict
            else None,
            mechanism_year_collection=None,
        )

    yield create_combined_measure
