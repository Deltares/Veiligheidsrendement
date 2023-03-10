from vrtool.failure_mechanisms.general import GenericFailureMechanismInput
from vrtool.failure_mechanisms.mechanism_input import MechanismInput


class TestGenericFailureMechanismInput:
    def test_from_mechanism_input_creates_expected_input(self):
        # Setup
        beta_table = {1: 0.1, 2: 0.2, 3: 0.3}
        mechanism_input = MechanismInput("")
        mechanism_input.input["beta"] = beta_table

        # Call
        generic_failure_mechanism_input = (
            GenericFailureMechanismInput.from_mechanism_input(mechanism_input)
        )

        # Assert
        assert generic_failure_mechanism_input.time_grid == list(beta_table.keys())
        assert generic_failure_mechanism_input.beta_grid == list(beta_table.values())
