from vrtool.failure_mechanisms.overflow.overflow import Overflow
from vrtool.failure_mechanisms.overflow.overflow_simple_input import OverflowSimpleInput


class OverflowSimple:
    def calculate(
        mechanism_input: OverflowSimpleInput,
    ) -> tuple[float, float]:
        """
        Calculates the overflow with a simple approximation.
        Args:
            mechanism_input (OverflowSimpleInput): The input specific for this mechanism.
        Returns:
            Tuple[float, float]: A tuple with the reliability and the probability of failure.
        """

        return Overflow.overflow_simple(
            mechanism_input.corrected_crest_height,
            mechanism_input.q_crest,
            mechanism_input.h_c,
            mechanism_input.q_c,
            mechanism_input.beta,
        )
