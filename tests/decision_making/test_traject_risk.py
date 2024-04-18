import numpy as np
import pytest

from vrtool.common.enums.mechanism_enum import MechanismEnum
from vrtool.decision_making.traject_risk import TrajectRisk

_MECHANISMS = [
    MechanismEnum.OVERFLOW,
    MechanismEnum.REVETMENT,
    MechanismEnum.STABILITY_INNER,
    MechanismEnum.PIPING,
]


class TestTrajectRisk:
    def _get_traject_risk(self) -> TrajectRisk:

        _initial_probabilities = [0.2, 0.3, 0.4, 0.5]
        _section1_probabilities = [
            _initial_probabilities,
            [p * 0.5 for p in _initial_probabilities],
            [p * 0.4 for p in _initial_probabilities],
        ]
        _section2_probabilities = [
            [p * 1.5 for p in _initial_probabilities],
            [p * 1.1 for p in _initial_probabilities],
            [p * 1.2 for p in _initial_probabilities],
        ]
        _probability_of_failure = {
            _mechanism.name: np.array(
                [_section1_probabilities, _section2_probabilities]
            )
            for _mechanism in _MECHANISMS
        }
        _annual_damage = np.array([1.0, 2.0, 3.0, 4.0])
        return TrajectRisk(_probability_of_failure, _annual_damage)

    def test_initialize(self):
        # 1. Define test data
        _measure_probabilities = [0.1, 0.2, 0.3, 0.4]
        _section_probabilities = [_measure_probabilities, _measure_probabilities]

        _probability_of_failure = {
            MechanismEnum.OVERFLOW.name: np.array(
                [_section_probabilities],
            )
        }
        _annual_damage = np.array([1.0, 2.0, 3.0, 4.0])

        # 2. Run test
        _tr = TrajectRisk(_probability_of_failure, _annual_damage)

        # 3. Verify expectations
        assert isinstance(_tr, TrajectRisk)
        assert np.array_equal(
            _tr.probability_of_failure[MechanismEnum.OVERFLOW],
            _probability_of_failure[MechanismEnum.OVERFLOW.name],
        )
        assert np.array_equal(_tr.annual_damage, _annual_damage)

    def test_get_initial_probabilities_copy(self):
        # 1. Define test data
        _tr = self._get_traject_risk()
        _mechanism = _MECHANISMS[0]

        # 2. Run test
        _init_probs_dict = _tr.get_initial_probabilities(
            [_mechanism, MechanismEnum.INVALID]
        )

        # 3. Verify expectations
        assert isinstance(_init_probs_dict, dict)
        assert len(_init_probs_dict) == 2
        assert _mechanism.name in _init_probs_dict
        assert np.array_equal(
            _init_probs_dict[_mechanism.name],
            _tr.probability_of_failure[_mechanism][:, 0, :],
        )

    @pytest.mark.parametrize(
        "mechanism, result",
        [
            pytest.param(MechanismEnum.OVERFLOW, 10.0, id="Overflow"),
            pytest.param(MechanismEnum.REVETMENT, 10.0, id="Revetment"),
        ],
    )
    def test_get_mechanism_risk(self, mechanism: MechanismEnum, result: float):
        # 1. Define test data
        _tr = self._get_traject_risk()

        # 2. Run test
        _mech_risk = _tr.get_mechanism_risk(mechanism)

        # 3. Verify expectations
        assert np.sum(_mech_risk) == pytest.approx(result)

    def test_get_independent_risk(self):
        # 1. Define test data
        _tr = self._get_traject_risk()

        # 2. Run test
        _independent_risk = _tr.get_independent_risk()

        # 3. Verify expectations
        assert np.sum(_independent_risk) == pytest.approx(14.475)

    def test_get_total_risk(self):
        # 1. Define test data
        _tr = self._get_traject_risk()

        # 2. Run test
        _init_risk = _tr.get_total_risk()

        # 3. Verify expectations
        assert _init_risk == pytest.approx(26.475)

    @pytest.mark.parametrize(
        "measure, result",
        [
            pytest.param((0, 0, 0), 26.475, id="No measure"),
            pytest.param((0, 1, 0), 26.475, id="Sh section 0"),
            pytest.param((0, 0, 1), 23.75, id="Sg section 0"),
            pytest.param((0, 1, 1), 23.75, id="Sh and Sg section 0"),
            pytest.param((1, 1, 1), 21.843, id="Sh and Sg section 1"),
        ],
    )
    def test_get_total_risk_for_measure(
        self, measure: tuple[int, int, int], result: float
    ):
        # 1. Define test data
        _tr = self._get_traject_risk()

        # 2. Run test
        _measure_risk = _tr.get_total_risk_for_measure(measure)

        # 3. Verify expectations
        assert _measure_risk == pytest.approx(result)
