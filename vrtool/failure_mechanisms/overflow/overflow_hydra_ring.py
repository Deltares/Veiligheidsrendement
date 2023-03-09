from vrtool.failure_mechanisms.overflow.overflow_functions import overflow_hring
from vrtool.failure_mechanisms.overflow.overflow_hydra_ring_input import (
    OverflowHydraRingInput,
)


class OverflowHydraRing:
    def calculate(
        mechanism_input: OverflowHydraRingInput, year: int, initial_year: int
    ) -> tuple[float, float]:
        """
        Calculates the reliability and safety factor.

        Args:
            mechanism_input (OverflowHydraRingInput): The input to perform the calculation with.
            year (int): The year to calculate the reliability and safety factor for.
            initial_year (int): The initial year.

        Returns:
            tuple[float, float]: A tuple with the calculated reliability and safety factor.
        """

        input = dict(
            h_crest=mechanism_input.h_crest,
            d_crest=mechanism_input.d_crest,
            hc_beta=mechanism_input.hc_beta,
        )

        return overflow_hring(input, year, initial_year)
