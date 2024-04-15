import numpy as np

from vrtool.common.enums.mechanism_enum import MechanismEnum
from vrtool.decision_making.strategy_output.mechanism_probabilities import (
    MechanismProbabilities,
)


class TestMechanismProbabilities:
    def test_initialize_from_strategy_input(self):
        # 1. Define test data
        _mech = MechanismEnum.OVERFLOW
        _prob_section1 = [0.1, 0.2, 0.3, 0.4]
        _prob_section2 = [0.5, 0.6, 0.7, 0.8]
        _section_probabilities = np.array([_prob_section1, _prob_section2])
        _mechanism_probabilities = np.array(
            [_section_probabilities, _section_probabilities]
        )
        _sh_idx = 0
        _sg_idx = 1

        # 2. Run test
        _mp = MechanismProbabilities.from_strategy_input(
            _mech, _mechanism_probabilities, _sh_idx, _sg_idx
        )

        # 3. Verify expectations
        assert isinstance(_mp, MechanismProbabilities)
        assert _mp.mechanism == _mech
        assert len(_mp.section_probabilities) == 2
        assert _mp.section_probabilities[0].year_probabilities == _prob_section1
