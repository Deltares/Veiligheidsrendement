from vrtool.failure_mechanisms.overflow.overflow_functions import overflow_simple
from vrtool.failure_mechanisms.overflow.overflow_simple_input import OverflowSimpleInput

from vrtool.flood_defence_system.load_input import LoadInput


class OverflowSimple:
    def calculate(
        mechanism_input: OverflowSimpleInput, year: int, load: LoadInput
    ) -> tuple[float, float]:
        """
        Calculates the overflow with a simple approximation.
        Args:
            mechanism_input (OverflowSimpleInput): The input specific for this mechanism.
        Returns:
            Tuple[float, float]: A tuple with the reliability and the probability of failure.
        """

        # climate change included, including a factor for HBN
        if hasattr(load, "dist_change"):
            corrected_crest_height = (
                mechanism_input.h_crest
                - (mechanism_input.dhc_t + (load.dist_change * load.HBN_factor)) * year
            )
        else:
            corrected_crest_height = mechanism_input.h_crest - (
                mechanism_input.dhc_t * year
            )

        return overflow_simple(
            corrected_crest_height,
            mechanism_input.q_crest,
            mechanism_input.h_c,
            mechanism_input.q_c,
            mechanism_input.beta,
        )
