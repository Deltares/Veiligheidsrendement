import numpy as np
import pytest

from vrtool.common.enums.mechanism_enum import MechanismEnum
from vrtool.decision_making.strategy_output.traject_probabilities import (
    TrajectProbabilities,
)


class TestTrajectProbabilities:
    class MockMechanismProbabilities:
        mechanism: MechanismEnum
        section_probabilities: np.ndarray

        def __init__(self, mechanism, probabilities):
            self.mechanism = mechanism
            self.section_probabilities = probabilities

        def get_probabilities(self):
            return self.section_probabilities

    def test_initialize_from_strategy_input(self):
        # 1. Define test data
        _prob_section1 = [0.1, 0.2, 0.3, 0.4]
        _prob_section2 = [0.5, 0.6, 0.7, 0.8]
        _section_probabilities = np.array([_prob_section1, _prob_section2])
        _mechanism_probabilities = np.array(
            [_section_probabilities, _section_probabilities]
        )
        _prob_failure_dict = {
            MechanismEnum.OVERFLOW.name: _mechanism_probabilities,
            MechanismEnum.REVETMENT.name: _mechanism_probabilities,
            MechanismEnum.PIPING.name: _mechanism_probabilities,
            MechanismEnum.INVALID.name: _mechanism_probabilities,
        }
        _damage = np.array([1.0, 2.0, 3.0, 4.0])
        _mechanisms = [
            MechanismEnum.OVERFLOW,
            MechanismEnum.REVETMENT,
            MechanismEnum.STABILITY_INNER,
        ]
        _sh_idx = 0
        _sg_idx = 1

        # 2. Run test
        _tp = TrajectProbabilities.from_strategy_input(
            _prob_failure_dict, _damage, _mechanisms, _sh_idx, _sg_idx
        )

        # 3. Verify expectations
        assert isinstance(_tp, TrajectProbabilities)
        assert len(_tp.mechanism_probabilities) == 2
        assert all(
            x == y
            for x, y in zip(
                _tp.mechanisms, [MechanismEnum.OVERFLOW, MechanismEnum.REVETMENT]
            )
        )
        assert np.array_equal(_tp.annual_damage, _damage)

    def test_combine_probabilities(self):
        # 1. Define test data
        _mechanism_probabilities = [
            self.MockMechanismProbabilities(
                MechanismEnum.OVERFLOW, np.array([0.01, 0.02, 0.03, 0.04])
            ),
            self.MockMechanismProbabilities(
                MechanismEnum.PIPING, np.array([0.05, 0.06, 0.07, 0.08])
            ),
            self.MockMechanismProbabilities(
                MechanismEnum.STABILITY_INNER, np.array([0.09, 0.10, 0.11, 0.12])
            ),
        ]
        _annual_damage = np.array([1.0, 2.0, 3.0, 4.0])
        _mechanisms = [
            MechanismEnum.STABILITY_INNER,
            MechanismEnum.PIPING,
            MechanismEnum.INVALID,
        ]
        _tp = TrajectProbabilities(
            mechanism_probabilities=_mechanism_probabilities,
            annual_damage=_annual_damage,
        )

        # 2. Run test
        _combined_probabilities = _tp.combine_probabilities(_mechanisms)

        # 3. Verify expectations
        assert all(
            np.isclose(
                _combined_probabilities,
                np.array([0.1355, 0.1540, 0.1723, 0.1904]),
                atol=1e-8,
            )
        )

    def test_get_total_risk(self):

        # 1. Define test data
        _mechanism_probabilities = [
            self.MockMechanismProbabilities(
                MechanismEnum.OVERFLOW, np.array([[0.1, 0.2, 0.3, 0.4]])
            ),
            self.MockMechanismProbabilities(
                MechanismEnum.REVETMENT, np.array([[0.5, 0.6, 0.7, 0.8]])
            ),
            self.MockMechanismProbabilities(
                MechanismEnum.STABILITY_INNER, np.array([[0.01, 0.02, 0.02, 0.04]])
            ),
            self.MockMechanismProbabilities(
                MechanismEnum.PIPING, np.array([[0.05, 0.06, 0.07, 0.08]])
            ),
        ]
        _damage = np.array([1.0, 2.0, 3.0, 4.0])
        _tp = TrajectProbabilities(
            mechanism_probabilities=_mechanism_probabilities,
            annual_damage=_damage,
        )

        # 2. Run test
        _result = _tp.get_total_risk()

        # 3. Verify expectations
        assert _result == pytest.approx(10.9501)
