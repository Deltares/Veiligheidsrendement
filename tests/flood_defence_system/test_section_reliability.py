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
from vrtool.flood_defence_system.section_reliability import (
    BETA_THRESHOLD,
    SectionReliability,
)
from vrtool.probabilistic_tools.probabilistic_functions import beta_to_pf, pf_to_beta


class TestSectionReliability:
    def test_init_sets_properties(self):
        # Call
        _section_reliability = SectionReliability()

        # Assert
        assert (
            len(_section_reliability.failure_mechanisms.get_available_mechanisms()) == 0
        )
        assert _section_reliability.get_reliabilities() == {}
        assert _section_reliability.get_reliabilities_for_mechanisms() == {}

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
        _sr_left.set_reliabilities({0: 0.1, 10: 0.2})
        _sr_left.set_reliabilities_for_mechanism(
            MechanismEnum.PIPING, {0: 0.05, 10: 0.1}
        )

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
        _sr_left.set_reliabilities({0: 0.1, 10: 0.2})
        _sr_left.set_reliabilities_for_mechanism(
            MechanismEnum.PIPING, {0: 0.05, 10: 0.1}
        )

        _sr_right = SectionReliability()
        _sr_right.set_reliabilities(section_pf)
        for _mech, _data in mechanism_pf.items():
            _sr_right.set_reliabilities_for_mechanism(_mech, _data)

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
        assert _section_reliability.get_reliabilities() == {}
        assert _section_reliability.get_reliabilities_for_mechanisms() == {}

        # 2. Run test.
        _section_reliability.calculate_section_reliability(
            _cross_sectional_requirements
        )

        # 3. Verify expectations.
        for _year, _reliability in _expected_result.items():
            assert _section_reliability.get_reliability_for_year(
                _year
            ) == pytest.approx(_reliability, abs=1e-6)

    def test_set_reliabilities(self):
        # 1. Define test data.
        _section_reliability = SectionReliability()
        _expected_reliabilities = {0: 0.1, 10: 0.2, 20: 0.3}
        _section_reliability.set_reliabilities(_expected_reliabilities)

        # 2. Run test.
        _reliabilities = _section_reliability.get_reliabilities()

        # 3. Verify expectations.
        assert _reliabilities == _expected_reliabilities

    def test_set_reliability_for_year(self):
        # 1. Define test data.
        _section_reliability = SectionReliability()
        _expected_reliabilities = {0: 0.1, 10: 0.2}
        for _year, _pf in _expected_reliabilities.items():
            _section_reliability.set_reliability_for_year(_year, _pf)

        # 2. Run test.
        _reliabilities = _section_reliability.get_reliabilities()

        # 3. Verify expectations.
        assert _reliabilities == _expected_reliabilities

    def test_set_reliabilities_for_mechanism(self):
        # 1. Define test data.
        _section_reliability = SectionReliability()
        _expected_reliabilities = {0: 0.05, 10: 0.1}
        _section_reliability.set_reliabilities_for_mechanism(
            MechanismEnum.PIPING, _expected_reliabilities
        )

        # 2. Run test.
        _reliabilities = _section_reliability.get_reliabilities_for_mechanism(
            MechanismEnum.PIPING
        )

        # 3. Verify expectations.
        assert _reliabilities == _expected_reliabilities

    @pytest.mark.parametrize(
        "beta, expected_beta",
        [
            pytest.param(7.0, 7.0, id="beta below threshold"),
            pytest.param(9.0, BETA_THRESHOLD, id="beta above threshold"),
        ],
    )
    def test_set_reliability_for_mechanism_year(
        self, beta: float, expected_beta: float
    ):
        # 1. Define test data.
        _mechanism = MechanismEnum.PIPING
        _section_reliability = SectionReliability()
        _expected_reliabilities = {0: 0.05, 10: beta_to_pf(beta)}
        for _year, _pf in _expected_reliabilities.items():
            _section_reliability.set_reliability_for_mechanism_year(
                _mechanism, _year, _pf
            )

        # 2. Run test.
        _beta_10 = pf_to_beta(
            _section_reliability.get_reliability_for_mechanism_year(_mechanism, 10)
        )

        # 3. Verify expectations.
        assert _beta_10 == pytest.approx(expected_beta)
