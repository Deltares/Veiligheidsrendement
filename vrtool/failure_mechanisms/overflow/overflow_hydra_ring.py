from vrtool.failure_mechanisms.overflow.overflow_hydra_ring_input import (
    OverflowHydraRingInput,
)
from vrtool.failure_mechanisms.overflow.overflow_functions import (
    calculate_overflow_hydra_ring_assessment,
)


class OverflowHydraRing:
    def calculate(
        mechanism_input: OverflowHydraRingInput, year: int, initial_year: int
    ) -> tuple[float, float]:
        """
        Calculates the reliability and the probability of failure.

        Args:
            mechanism_input (OverflowHydraRingInput): The input to perform the calculation with.
            year (int): The year to calculate the reliability and the probability of failure for.
            initial_year (int): The initial year.

        Returns:
            tuple[float, float]: A tuple with the calculated reliability and the probability of failure.
        """

        return calculate_overflow_hydra_ring_assessment(
            year,
            initial_year,
            mechanism_input.h_crest,
            mechanism_input.d_crest,
            mechanism_input.hc_beta,
        )
