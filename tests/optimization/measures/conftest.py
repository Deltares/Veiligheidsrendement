from typing import Callable, Iterable

import pytest

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
