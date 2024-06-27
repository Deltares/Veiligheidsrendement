from dataclasses import dataclass
from typing import Callable, Iterable, Iterator

import pytest

from vrtool.common.enums.measure_type_enum import MeasureTypeEnum
from vrtool.common.enums.mechanism_enum import MechanismEnum
from vrtool.optimization.measures.combined_measures.combined_measure_base import (
    CombinedMeasureBase,
)
from vrtool.optimization.measures.combined_measures.combined_measure_factory import (
    CombinedMeasureFactory,
)
from vrtool.optimization.measures.combined_measures.sg_combined_measure import (
    SgCombinedMeasure,
)
from vrtool.optimization.measures.combined_measures.sh_combined_measure import (
    ShCombinedMeasure,
)
from vrtool.optimization.measures.combined_measures.shsg_combined_measure import (
    ShSgCombinedMeasure,
)
from vrtool.optimization.measures.measure_as_input_protocol import (
    MeasureAsInputProtocol,
)
from vrtool.optimization.measures.mechanism_per_year import MechanismPerYear
from vrtool.optimization.measures.mechanism_per_year_probability_collection import (
    MechanismPerYearProbabilityCollection,
)
from vrtool.optimization.measures.sh_measure import ShMeasure


class TestCombinedMeasureBase:
    @pytest.fixture(name="mocked_measure")
    def _get_valid_measure(
        self, mocked_measure_as_input: type[MeasureAsInputProtocol]
    ) -> Iterator[Callable[[MeasureTypeEnum, int], MeasureAsInputProtocol]]:
        def create_mocked_combined_measure(
            measure_type: MeasureTypeEnum,
            measure_result_id: int,
        ) -> MeasureAsInputProtocol:
            return mocked_measure_as_input(
                **(
                    dict(
                        measure_type=measure_type,
                        measure_result_id=measure_result_id,
                        mechanism_year_collection=self._get_valid_probability_collection(
                            MechanismEnum.OVERFLOW
                        ),
                    )
                )
            )

        yield create_mocked_combined_measure

    def _get_valid_probability_collection(
        self, mechanism: MechanismEnum
    ) -> MechanismPerYearProbabilityCollection:
        _mech_per_year = MechanismPerYear(mechanism=mechanism, year=0, probability=0.5)
        return MechanismPerYearProbabilityCollection(probabilities=[_mech_per_year])

    @pytest.mark.parametrize(
        "measure_type, expected",
        [pytest.param(MeasureTypeEnum.CUSTOM, False)]
        + [
            pytest.param(_measure_type, True)
            for _measure_type in MeasureTypeEnum
            if _measure_type not in (MeasureTypeEnum.INVALID, MeasureTypeEnum.CUSTOM)
        ],
    )
    def test_compares_to(
        self,
        measure_type: MeasureTypeEnum,
        expected: bool,
        mocked_measure: Callable[[MeasureTypeEnum, int], MeasureAsInputProtocol],
    ):
        # 1. Define test data
        _this_primary_measure_result_id = 1
        _other_primary_measure_result_id = 2
        _this_primary = mocked_measure(measure_type, _this_primary_measure_result_id)
        _other_primary = mocked_measure(measure_type, _other_primary_measure_result_id)

        _this_combination = CombinedMeasureBase(
            primary=_this_primary,
            mechanism_year_collection=self._get_valid_probability_collection(
                MechanismEnum.OVERFLOW
            ),
            sequence_nr=7,
        )
        _other_combination = CombinedMeasureBase(
            primary=_other_primary,
            mechanism_year_collection=self._get_valid_probability_collection(
                MechanismEnum.OVERFLOW
            ),
            sequence_nr=8,
        )

        # 2. Run test
        _result = _this_combination.compares_to(_other_combination)

        # 3. Verify expectations
        assert _result == expected

    @pytest.fixture(name="combined_measure_example_factory")
    def _get_combined_measure_example_fixture(
        self,
        combined_measure_factory: Callable[
            [type[CombinedMeasureBase], dict, dict], CombinedMeasureBase
        ],
    ) -> Iterable[Callable[[type[CombinedMeasureBase]], CombinedMeasureBase]]:
        def create_combined_measure(
            combined_measure_type: type[CombinedMeasureBase],
        ) -> CombinedMeasureBase:
            return combined_measure_factory(
                combined_measure_type,
                dict(cost=4.2, base_cost=2.2, year=20),
                dict(cost=6.7, base_cost=4.2, year=0),
            )

        yield create_combined_measure

    @pytest.mark.parametrize(
        "combined_measure_type, expected_result",
        [
            pytest.param(ShCombinedMeasure, 6.7105),
            pytest.param(SgCombinedMeasure, 6.7105),
            pytest.param(ShSgCombinedMeasure, 6.7105),
        ],
    )
    def test_lcc_for_combined_measure_types(
        self,
        combined_measure_type: type[CombinedMeasureBase],
        expected_result: float,
        combined_measure_example_factory: Callable[
            [type[CombinedMeasureBase]], CombinedMeasureBase
        ],
    ):
        # 1. Define test data.
        _combined_measure_example = combined_measure_example_factory(
            combined_measure_type
        )

        # 2. Run test.
        assert _combined_measure_example.lcc == pytest.approx(expected_result, 0.0001)
