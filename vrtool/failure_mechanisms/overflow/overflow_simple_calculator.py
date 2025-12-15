from vrtool.failure_mechanisms.failure_mechanism_calculator_protocol import (
    FailureMechanismCalculatorProtocol,
)
from vrtool.failure_mechanisms.overflow.overflow_functions import (
    calculate_overflow_simple_assessment,
)
from vrtool.failure_mechanisms.overflow.overflow_simple_input import OverflowSimpleInput


class OverflowSimpleCalculator(FailureMechanismCalculatorProtocol):
    def __init__(self, mechanism_input: OverflowSimpleInput) -> None:
        if not isinstance(mechanism_input, OverflowSimpleInput):
            raise ValueError(
                "Expected instance of a {}.".format(OverflowSimpleInput.__name__)
            )

        self._mechanism_input = mechanism_input

    def calculate(self, year: int) -> tuple[float, float]:
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
