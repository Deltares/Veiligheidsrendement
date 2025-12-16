import pandas as pd

from vrtool.failure_mechanisms.mechanism_input import MechanismInput
from vrtool.failure_mechanisms.overflow import OverflowHydraRingInput


class TestOverflowHydraRingInput:
    def test_from_mechanism_input_creates_expected_input(self):
        # Setup
        mechanism_input = MechanismInput("")
        mechanism_input.input_dict["h_crest"] = 0.1
        mechanism_input.input_dict["d_crest"] = 0.2
        mechanism_input.input_dict["hc_beta"] = pd.DataFrame(
            {"col1": [1, 2], "col2": [3, 4]}
        )

        # Call
        overflow_hydra_ring_input = OverflowHydraRingInput.from_mechanism_input(
            mechanism_input
        )

        # Assert
        assert (
            overflow_hydra_ring_input.h_crest == mechanism_input.input_dict["h_crest"]
        )
        assert (
            overflow_hydra_ring_input.d_crest == mechanism_input.input_dict["d_crest"]
        )
        assert overflow_hydra_ring_input.hc_beta.equals(
            mechanism_input.input_dict["hc_beta"]
        )
