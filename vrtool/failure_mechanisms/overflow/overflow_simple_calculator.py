from vrtool.failure_mechanisms.failure_mechanism_calculator_protocol import (
    FailureMechanismCalculatorProtocol,
)
from vrtool.failure_mechanisms.overflow.overflow_functions import (
    calculate_overflow_simple_assessment,
)
from vrtool.failure_mechanisms.overflow.overflow_simple_input import OverflowSimpleInput

from vrtool.flood_defence_system.load_input import LoadInput


class OverflowSimpleCalculator(FailureMechanismCalculatorProtocol):
    def __init__(self, mechanism_input: OverflowSimpleInput, load: LoadInput) -> None:
        if not isinstance(mechanism_input, OverflowSimpleInput):
            raise ValueError(
                "Expected instance of a {}.".format(OverflowSimpleInput.__name__)
            )

        if not isinstance(load, LoadInput):
            raise ValueError("Expected instance of a {}.".format(LoadInput.__name__))

        self._mechanism_input = mechanism_input
        self._load = load

    def calculate(self, year: int) -> tuple[float, float]:
        # climate change included, including a factor for HBN
        if hasattr(self._load, "dist_change"):
            corrected_crest_height = (
                self._mechanism_input.h_crest
                - (
                    self._mechanism_input.dhc_t
                    + (self._load.dist_change * self._load.HBN_factor)
                )
                * year
            )
        else:
            corrected_crest_height = self._mechanism_input.h_crest - (
                self._mechanism_input.dhc_t * year
            )

        return calculate_overflow_simple_assessment(
            corrected_crest_height,
            self._mechanism_input.q_crest,
            self._mechanism_input.h_c,
            self._mechanism_input.q_c,
            self._mechanism_input.beta,
        )
