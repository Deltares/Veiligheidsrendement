from typing import Callable, Iterator

import pandas as pd
import pytest

from vrtool.common.enums.computation_type_enum import ComputationTypeEnum
from vrtool.common.enums.mechanism_enum import MechanismEnum
from vrtool.flood_defence_system.cross_sectional_requirements import (
    CrossSectionalRequirements,
)
from vrtool.flood_defence_system.mechanism_reliability_collection import (
    MechanismReliabilityCollection,
)
from vrtool.flood_defence_system.section_reliability import SectionReliability
from vrtool.probabilistic_tools.probabilistic_functions import beta_to_pf


class TestSectionReliability:
    def test_init_sets_properties(self):
        # Call
        _section_reliability = SectionReliability()

        # Assert
        assert (
            len(_section_reliability.failure_mechanisms.get_available_mechanisms()) == 0
        )
        assert _section_reliability.section_pf == {}
        assert _section_reliability.mechanism_pf == {}

    @pytest.mark.parametrize(
        "other",
        [
            pytest.param(None, id="None"),
            pytest.param("NotASectionReliability", id="Different type"),
            pytest.param(SectionReliability(), id="Empty object"),
        ],
    )
    def test_equality_basic_false_cases(self, other: SectionReliability):
        # 1. Define test data.
        _sr_left = SectionReliability()
        _sr_left.section_pf = {0: 0.1, 10: 0.2}
        _sr_left.mechanism_pf = {MechanismEnum.PIPING: {0: 0.05, 10: 0.1}}

        # 2. Run test.
        _result = _sr_left == other

        # 3. Verify expectations.
        assert _result is False

    @pytest.mark.parametrize(
        "section_pf, mechanism_pf, expected",
        [
            pytest.param(
                {0: 0.1, 10: 0.2},
                {MechanismEnum.PIPING: {0: 0.05, 10: 0.1}},
                True,
                id="Matching simple case",
            ),
            pytest.param(
                {0: 0.1, 10: 0.2},
                {MechanismEnum.OVERFLOW: {0: 0.05, 10: 0.1}},
                False,
                id="Different mechanism",
            ),
            pytest.param(
                {0: 0.1, 20: 0.2},
                {MechanismEnum.PIPING: {0: 0.05, 20: 0.1}},
                False,
                id="Different year",
            ),
            pytest.param(
                {0: 0.1, 10: 0.3},
                {MechanismEnum.PIPING: {0: 0.05, 10: 0.1}},
                False,
                id="Different section_pf value",
            ),
            pytest.param(
                {0: 0.1, 10: 0.2},
                {MechanismEnum.PIPING: {0: 0.05, 10: 0.2}},
                False,
                id="Different mechanism_pf value",
            ),
        ],
    )
    def test_equality_detailed_cases(
        self,
        section_pf: dict[int, float],
        mechanism_pf: dict[MechanismEnum, dict[int, float]],
        expected: bool,
    ):
        # 1. Define test data.
        _sr_left = SectionReliability()
        _sr_left.section_pf = {0: 0.1, 10: 0.2}
        _sr_left.mechanism_pf = {MechanismEnum.PIPING: {0: 0.05, 10: 0.1}}

        _sr_right = SectionReliability()
        _sr_right.section_pf = section_pf
        _sr_right.mechanism_pf = mechanism_pf

        # 2. Run test.
        _result = _sr_left == _sr_right

        # 3. Verify expectations.
        assert _result is expected

    @pytest.fixture(name="section_reliability_builder")
    def _get_section_reliability_builder_fixture(
        self,
    ) -> Iterator[Callable[[MechanismEnum], SectionReliability]]:
        _years = [0, 10]

        def _get_mrc(mechanism: MechanismEnum) -> MechanismReliabilityCollection:
            _mrc = MechanismReliabilityCollection(
                mechanism=mechanism,
                computation_type=ComputationTypeEnum.SIMPLE,
                computation_years=_years,
                t_0=_years[0],
                measure_year=5,
            )
            for _yt in _years:
                _mrc.Reliability[_yt].Pf = 0.42 / (100 * max(1, _yt))
            return _mrc

        def build_section_reliability_for_mechanism(
            mechanism: MechanismEnum,
        ) -> SectionReliability:
            _section_reliability = SectionReliability()
            _section_reliability.failure_mechanisms.add_failure_mechanism_reliability_collection(
                _get_mrc(mechanism)
            )
            return _section_reliability

        yield build_section_reliability_for_mechanism

    @pytest.mark.parametrize(
        "mechanism, expected_values",
        [
            pytest.param(
                MechanismEnum.PIPING, [0.791935, 1.981413], id=str(MechanismEnum.PIPING)
            ),
            pytest.param(
                MechanismEnum.STABILITY_INNER,
                [0.561045, 1.835016],
                id=str(MechanismEnum.STABILITY_INNER),
            ),
        ],
    )
    def test_calculate_section_reliability(
        self,
        mechanism: MechanismEnum,
        expected_values: list[float],
        section_reliability_builder: Callable[[MechanismEnum], SectionReliability],
    ):
        # 1. Define test data.
        _expected_result = {
            0: beta_to_pf(expected_values[0]),
            10: beta_to_pf(expected_values[1]),
        }
        _cross_sectional_requirements = CrossSectionalRequirements(
            dike_section_length=42,
            dike_section_a_piping=1.5,
            dike_section_a_stability_inner=2.3,
            dike_traject_b_piping=1.1,
            dike_traject_b_stability_inner=1.2,
        )
        _section_reliability = section_reliability_builder(mechanism)
        assert isinstance(_section_reliability, SectionReliability)
        assert _section_reliability.section_pf == {}
        assert _section_reliability.mechanism_pf == {}

        # 2. Run test.
        _section_reliability.calculate_section_reliability(
            _cross_sectional_requirements
        )

        # 3. Verify expectations.
        for _key, _val in _expected_result.items():
            assert _section_reliability.section_pf.get(_key) == pytest.approx(
                _val, abs=1e-6
            )
