import numpy as np

from vrtool.failure_mechanisms.mechanism_input import MechanismInput
from vrtool.failure_mechanisms.overflow import OverflowSimpleInput


class TestOverFlowSimpleInput:
    def test_from_mechanism_input_creates_expected_input(self):
        # Setup
        mechanism_input = MechanismInput("")
        mechanism_input.input_dict["dhc(t)"] = np.array(0.1)
        mechanism_input.input_dict["h_crest"] = np.array(0.2)
        mechanism_input.input_dict["q_crest"] = np.array(0.3)
        mechanism_input.input_dict["h_c"] = np.array(0.4)
        mechanism_input.input_dict["q_c"] = np.array(0.5)
        mechanism_input.input_dict["beta"] = np.array(0.6)

        # Call
        overflow_simple_input = OverflowSimpleInput.from_mechanism_input(
            mechanism_input
        )

        # Assert
        assert overflow_simple_input.dhc_t == mechanism_input.input_dict["dhc(t)"]
        assert overflow_simple_input.h_crest == mechanism_input.input_dict["h_crest"]
        assert overflow_simple_input.q_crest == mechanism_input.input_dict["q_crest"]
        assert overflow_simple_input.h_c == mechanism_input.input_dict["h_c"]
        assert overflow_simple_input.q_c == mechanism_input.input_dict["q_c"]
        assert overflow_simple_input.beta == mechanism_input.input_dict["beta"]
