import numpy as np
import pytest

from vrtool.common.enums.mechanism_enum import MechanismEnum
from vrtool.decision_making.strategy_output.mechanism_probabilities import (
    MechanismProbabilities,
)
from vrtool.decision_making.strategy_output.section_year_probabilities import (
    SectionYearProbabilities,
)


class TestMechanismProbabilities:
    def test_initialize_from_strategy_input(self):
        # 1. Define test data
        _mech = MechanismEnum.OVERFLOW
        _prob_section1 = np.array([0.1, 0.2, 0.3, 0.4])
        _prob_section2 = np.array([0.5, 0.6, 0.7, 0.8])
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
        assert np.array_equal(
            _mp.section_probabilities[0].probabilities, _prob_section1
        )

    @pytest.mark.parametrize(
        "mechanism",
        [
            pytest.param(MechanismEnum.OVERFLOW, id="Overflow"),
            pytest.param(MechanismEnum.REVETMENT, id="Revetment"),
        ],
    )
    def test_get_dependent_probabilities(self, mechanism: MechanismEnum):
        # 1. Define test data
        _prob_section1 = np.array([0.1, 0.2, 0.3, 0.4])
        _prob_section2 = np.array([0.5, 0.6, 0.7, 0.8])
        _mp = MechanismProbabilities(
            mechanism=mechanism,
            section_probabilities=[
                SectionYearProbabilities(probabilities=_prob_section1),
                SectionYearProbabilities(probabilities=_prob_section2),
            ],
        )

        # 2. Run test
        _result = _mp.get_probabilities()

        # 3. Verify expectations
        assert np.array_equal(_result, _prob_section2)

    @pytest.mark.parametrize(
        "mechanism",
        [
            pytest.param(MechanismEnum.STABILITY_INNER, id="Stability Inner"),
            pytest.param(MechanismEnum.PIPING, id="Piping"),
        ],
    )
    def test_get_independent_probabilities(self, mechanism: MechanismEnum):
        # 1. Define test data
        _prob_section1 = np.array([0.1, 0.2, 0.3, 0.4])
        _prob_section2 = np.array([0.5, 0.6, 0.7, 0.8])
        _mp = MechanismProbabilities(
            mechanism=mechanism,
            section_probabilities=[
                SectionYearProbabilities(probabilities=_prob_section1),
                SectionYearProbabilities(probabilities=_prob_section2),
            ],
        )

        # 2. Run test
        _result = _mp.get_probabilities()

        # 3. Verify expectations
        assert np.array_equal(_result, _prob_section1 + _prob_section2)
