from vrtool.failure_mechanisms.overflow import OverflowSimpleInput
from vrtool.failure_mechanisms.mechanism_input import MechanismInput
import numpy as np


class TestOverFlowSimpleInput:
    def test_from_mechanism_input_creates_expected_input(self):
        mechanism_input = MechanismInput("")
        mechanism_input.input["dhc(t)"] = np.array(0.1)
        mechanism_input.input["h_crest"] = np.array(0.2)
        mechanism_input.input["q_crest"] = np.array(0.3)
        mechanism_input.input["h_c"] = np.array(0.4)
        mechanism_input.input["q_c"] = np.array(0.5)
        mechanism_input.input["beta"] = np.array(0.6)

        overflow_simple_input = OverflowSimpleInput.from_mechanism_input(
            mechanism_input
        )

        assert overflow_simple_input.dhc_t == mechanism_input.input["dhc(t)"]
        assert overflow_simple_input.h_crest == mechanism_input.input["h_crest"]
        assert overflow_simple_input.q_crest == mechanism_input.input["q_crest"]
        assert overflow_simple_input.h_c == mechanism_input.input["h_c"]
        assert overflow_simple_input.q_c == mechanism_input.input["q_c"]
        assert overflow_simple_input.beta == mechanism_input.input["beta"]
