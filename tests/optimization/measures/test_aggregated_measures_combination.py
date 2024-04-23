from dataclasses import dataclass

import pytest

from vrtool.optimization.measures.aggregated_measures_combination import (
    AggregatedMeasureCombination,
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
        "sh_meas_id, sh_prim_id, sh_id_match",
        [
            pytest.param(1, 1, True, id="Sh id match"),
            pytest.param(1, 2, False, id="Sh id don't match"),
        ],
    )
    @pytest.mark.parametrize(
        "sh_meas_year, sh_prim_year, sh_year_match",
        [
            pytest.param(1, 1, True, id="Sh year match"),
            pytest.param(1, 2, False, id="Sh year don't match"),
        ],
    )
    @pytest.mark.parametrize(
        "sg_meas_id, sg_prim_id, sg_id_match",
        [
            pytest.param(1, 1, True, id="Sg id match"),
            pytest.param(1, 2, False, id="Sg id don't match"),
        ],
    )
    @pytest.mark.parametrize(
        "sg_meas_year, sg_prim_year, sg_year_match",
        [
            pytest.param(1, 1, True, id="Sg year match"),
            pytest.param(1, 2, False, id="Sg year don't match"),
        ],
    )
    def test_check_primary_measure_result_id_and_year(
        self,
        sh_meas_id: int,
        sh_prim_id: int,
        sh_id_match: bool,
        sh_meas_year: int,
        sh_prim_year: int,
        sh_year_match: bool,
        sg_meas_id: int,
        sg_prim_id: int,
        sg_id_match: bool,
        sg_meas_year: int,
        sg_prim_year: int,
        sg_year_match: bool,
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
        _prim_sh = self.MockMeasure(measure_result_id=sh_prim_id, year=sh_prim_year)
        _prim_sg = self.MockMeasure(measure_result_id=sg_prim_id, year=sg_prim_year)

        # 2. Run test
        _result = _amc.check_primary_measure_result_id_and_year(
            primary_sh=_prim_sh,
            primary_sg=_prim_sg,
        )

        # 3. Verify expectations
        assert isinstance(_result, bool)
        assert _result == (
            sh_id_match and sh_year_match and sg_id_match and sg_year_match
        )

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
