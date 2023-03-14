from vrtool.failure_mechanisms.overflow import OverflowHydraRingInput
from vrtool.failure_mechanisms.mechanism_input import MechanismInput
import pandas as pd


class TestOverflowHydraRingInput:
    def test_from_mechanism_input_creates_expected_input(self):
        # Setup
        mechanism_input = MechanismInput("")
        mechanism_input.input["h_crest"] = 0.1
        mechanism_input.input["d_crest"] = 0.2
        mechanism_input.input["hc_beta"] = pd.DataFrame(
            {"col1": [1, 2], "col2": [3, 4]}
        )

        # Call
        overflow_hydra_ring_input = OverflowHydraRingInput.from_mechanism_input(
            mechanism_input
        )

        # Assert
        assert overflow_hydra_ring_input.h_crest == mechanism_input.input["h_crest"]
        assert overflow_hydra_ring_input.d_crest == mechanism_input.input["d_crest"]
        assert overflow_hydra_ring_input.hc_beta.equals(
            mechanism_input.input["hc_beta"]
        )
