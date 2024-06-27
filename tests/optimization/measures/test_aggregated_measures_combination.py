from dataclasses import dataclass
from typing import Callable

import pytest

from vrtool.optimization.measures.aggregated_measures_combination import (
    AggregatedMeasureCombination,
)
from vrtool.optimization.measures.combined_measures.combined_measure_base import (
    CombinedMeasureBase,
)
from vrtool.optimization.measures.combined_measures.sg_combined_measure import (
    SgCombinedMeasure,
)
from vrtool.optimization.measures.combined_measures.sh_combined_measure import (
    ShCombinedMeasure,
)
from vrtool.optimization.measures.measure_as_input_protocol import (
    MeasureAsInputProtocol,
)


class TestAggregatedMeasuresCombination:
    @dataclass
    class MockCombinedMeasure:
        primary: MeasureAsInputProtocol = None
        sequence_nr: int = 0

    @dataclass
    class MockMeasure(MeasureAsInputProtocol):
        year: int = 0
        measure_result_id: int = 0

    def test_initizalize(self):
        # 1. Define test data
        _sh_combination = self.MockCombinedMeasure(sequence_nr=1)
        _sg_combination = self.MockCombinedMeasure(sequence_nr=2)

        # 2. Run test
        _amc = AggregatedMeasureCombination(
            sh_combination=_sh_combination,
            sg_combination=_sg_combination,
            measure_result_id=0,
            year=0,
        )

        # 3. Verify expectations
        assert isinstance(_amc, AggregatedMeasureCombination)

    @pytest.mark.parametrize(
        "sh_meas_id, sh_meas_year, sg_meas_id, sg_meas_year, expected",
        [
            pytest.param(1, 1, 1, 1, True, id="Sh/Sg id and year match"),
            pytest.param(2, 1, 1, 1, False, id="Sh id doesn't match"),
            pytest.param(1, 2, 1, 1, False, id="Sh year doesn't match"),
            pytest.param(1, 1, 2, 1, False, id="Sg id doesn't match"),
            pytest.param(1, 1, 1, 2, False, id="Sg year doesn't match"),
            pytest.param(2, 2, 2, 2, False, id="Sh/Sg id and year don't match"),
        ],
    )
    def test_check_primary_measure_result_id_and_year(
        self,
        sh_meas_id: int,
        sh_meas_year: int,
        sg_meas_id: int,
        sg_meas_year: int,
        expected: bool,
    ):
        # 1. Define test data
        _sh_meas = self.MockMeasure(measure_result_id=sh_meas_id, year=sh_meas_year)
        _sh_combination = self.MockCombinedMeasure(primary=_sh_meas, sequence_nr=1)
        _sg_meas = self.MockMeasure(measure_result_id=sg_meas_id, year=sg_meas_year)
        _sg_combination = self.MockCombinedMeasure(primary=_sg_meas, sequence_nr=2)
        _amc = AggregatedMeasureCombination(
            sh_combination=_sh_combination,
            sg_combination=_sg_combination,
            measure_result_id=0,
            year=0,
        )
        _prim_id = 1
        _prim_year = 1
        _prim_sh = self.MockMeasure(measure_result_id=_prim_id, year=_prim_year)
        _prim_sg = self.MockMeasure(measure_result_id=_prim_id, year=_prim_year)

        # 2. Run test
        _result = _amc.check_primary_measure_result_id_and_year(
            primary_sh=_prim_sh,
            primary_sg=_prim_sg,
        )

        # 3. Verify expectations
        assert isinstance(_result, bool)
        assert _result == expected

    def test_get_combination_idx(self):
        # 1. Define test data
        _sh_combination = self.MockCombinedMeasure(sequence_nr=1)
        _sg_combination = self.MockCombinedMeasure(sequence_nr=2)
        _aggregated_measure_combination = AggregatedMeasureCombination(
            sh_combination=_sh_combination,
            sg_combination=_sg_combination,
            measure_result_id=0,
            year=0,
        )

        # 2. Run test
        _sh_idx, _sg_idx = _aggregated_measure_combination.get_combination_idx()

        # 3. Verify expectations
        assert _sh_idx == 1
        assert _sg_idx == 2

    def test_lcc_given_aggregation_with_sh_and_sg_with_different_investment_years(
        self,
        combined_measure_factory_fixture: Callable[
            [type[CombinedMeasureBase], dict, dict], CombinedMeasureBase
        ],
    ):

        # 1. Define test data.
        _sh_combined_measure = combined_measure_factory_fixture(
            combined_measure_type=ShCombinedMeasure,
            primary_dict=dict(cost=4.2, base_cost=2.2, year=20),
            secondary_dict=dict(cost=6.7, base_cost=2.2, year=0),
        )

        _sg_combined_measure = combined_measure_factory_fixture(
            combined_measure_type=SgCombinedMeasure,
            primary_dict=dict(cost=2.4, base_cost=1.4, year=20),
            secondary_dict=dict(cost=4.6, base_cost=1.4, year=0),
        )

        # 2. Run test.
        _aggregated_measure = AggregatedMeasureCombination(
            sh_combination=_sh_combined_measure,
            sg_combination=_sg_combined_measure,
            measure_result_id=-1,
            year=0,
        )

        # 3. Verify expectations.
        assert _aggregated_measure.lcc == pytest.approx(11.3231, 0.0001)

    def test_lcc_given_aggregation_with_sh_with_different_investment_years(
        self,
        combined_measure_factory_fixture: Callable[
            [type[CombinedMeasureBase], dict, dict], CombinedMeasureBase
        ],
    ):
        # 1. Define test data.
        _sh_combined_measure = combined_measure_factory_fixture(
            combined_measure_type=ShCombinedMeasure,
            primary_dict=dict(cost=4.2, base_cost=4.2, year=20),
            secondary_dict=dict(cost=6.7, base_cost=4.2, year=0),
        )

        _sg_combined_measure = combined_measure_factory_fixture(
            combined_measure_type=SgCombinedMeasure,
            primary_dict=dict(cost=2.4, base_cost=2.4, year=20),
            secondary_dict=None,
        )

        # 2. Run test.
        _aggregated_measure = AggregatedMeasureCombination(
            sh_combination=_sh_combined_measure,
            sg_combination=_sg_combined_measure,
            measure_result_id=-1,
            year=0,
        )

        # 3. Verify expectations.
        assert _aggregated_measure.lcc == pytest.approx(6.7126, 0.0001)
