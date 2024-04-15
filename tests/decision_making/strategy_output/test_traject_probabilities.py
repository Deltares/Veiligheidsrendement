import numpy as np

from vrtool.common.enums.mechanism_enum import MechanismEnum
from vrtool.decision_making.strategy_output.traject_probabilities import (
    TrajectProbabilities,
)


class TestTrajectProbabilities:
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
            MechanismEnum.STABILITY_INNER.name: _mechanism_probabilities,
            MechanismEnum.PIPING.name: _mechanism_probabilities,
            MechanismEnum.INVALID.name: _mechanism_probabilities,
        }
        _damage_list = np.array([1.0, 2.0, 3.0, 4.0])
        _mechanisms = [
            MechanismEnum.OVERFLOW,
            MechanismEnum.REVETMENT,
            MechanismEnum.STABILITY_INNER,
            MechanismEnum.PIPING,
        ]
        _sh_idx = 0
        _sg_idx = 1

        # 2. Run test
        _tp = TrajectProbabilities.from_strategy_input(
            _prob_failure_dict, _damage_list, _mechanisms, _sh_idx, _sg_idx
        )

        # 3. Verify expectations
        assert isinstance(_tp, TrajectProbabilities)
        assert _tp.mechanisms == _mechanisms
        assert len(_tp.mechanism_prob) == 4
        assert _tp.annual_damage == _damage_list.tolist()
