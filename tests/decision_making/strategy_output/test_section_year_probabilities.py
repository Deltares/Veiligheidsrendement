import numpy as np

from vrtool.decision_making.strategy_output.section_year_probabilities import (
    SectionYearProbabilities,
)


class TestSectionYearProbabilities:
    def test_initialize_from_strategy_input(self):
        # 1. Define test data
        _probabilities = np.array([0.1, 0.2, 0.3, 0.4])

        # 2. Run test
        _syp = SectionYearProbabilities.from_strategy_input(_probabilities)

        # 3. Verify expectations
        assert isinstance(_syp, SectionYearProbabilities)
        assert np.array_equal(_syp.probabilities, _probabilities)

    def test_get_probabilities(self):
        # 1. Define test data
        _probabilities = np.array([0.1, 0.2, 0.3, 0.4])
        _syp = SectionYearProbabilities(probabilities=_probabilities)

        # 2. Run test
        _result = _syp.get_probabilities()

        # 3. Verify expectations
        assert isinstance(_result, np.ndarray)
        assert np.array_equal(_result, _probabilities)
