import numpy as np
import pytest

from vrtool.probabilistic_tools.combin_functions import CombinFunctions


class TestCombinFunctions:
    def testCombineProbabilities(self):
        a = [0.7, 0.8]
        b = [0.1, 0.2]
        dic = {"piping": np.array(a), "stability": np.array(b)}

        result = CombinFunctions.combine_probabilities(dic, ("stability", "piping"))

        expected0 = 1 - (1 - a[0]) * (1 - b[0])
        expected1 = 1 - (1 - a[1]) * (1 - b[1])
        assert len(result) == 2
        assert result[0] == pytest.approx(expected0, 1e-7)
        assert result[1] == pytest.approx(expected1, 1e-7)

    def testCombineProbabilitiesOneMechanismn(self):
        a = [0.7, 0.8]
        b = [0.1, 0.2]
        dic = {"piping": np.array(a), "stability": np.array(b)}

        result = CombinFunctions.combine_probabilities(dic, ("stability", "overflow"))

        assert len(result) == 2
        assert result[0] == b[0]
        assert result[1] == b[1]

    def testCombineProbabilitiesThreeMechanismns(self):
        a = [0.7, 0.8]
        b = [0.1, 0.2]
        c = [0.4, 0.5]
        dic = {"piping": np.array(a), "stability": np.array(b), "new": np.array(c)}

        result = CombinFunctions.combine_probabilities(
            dic, ("stability", "piping", "new")
        )

        expected0 = 1 - (1 - a[0]) * (1 - b[0]) * (1 - c[0])
        expected1 = 1 - (1 - a[1]) * (1 - b[1]) * (1 - c[1])
        assert len(result) == 2
        assert result[0] == pytest.approx(expected0, 1e-7)
        assert result[1] == pytest.approx(expected1, 1e-7)
