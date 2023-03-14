from vrtool.failure_mechanisms.overflow.overflow_hydra_ring_input import (
    OverflowHydraRingInput,
)
from vrtool.failure_mechanisms.overflow.overflow_functions import (
    calculate_overflow_hydra_ring_assessment,
)

from vrtool.failure_mechanisms.failure_mechanism_calculator_protocol import (
    FailureMechanismCalculatorProtocol,
)


class OverflowHydraRingCalculator(FailureMechanismCalculatorProtocol):
    def __init__(
        self, mechanism_input: OverflowHydraRingInput, initial_year: int
    ) -> None:
        if not isinstance(mechanism_input, OverflowHydraRingInput):
            raise ValueError(
                "Expected instance of a {}.".format(OverflowHydraRingInput.__name__)
            )

        if not isinstance(initial_year, int):
            raise ValueError("Expected instance of a {}.".format(int.__name__))

        self._mechanism_input = mechanism_input
        self._initial_year = initial_year

    def calculate(self, year: int) -> tuple[float, float]:
        return calculate_overflow_hydra_ring_assessment(
            year,
            self._initial_year,
            self._mechanism_input.h_crest,
            self._mechanism_input.d_crest,
            self._mechanism_input.hc_beta,
        )
