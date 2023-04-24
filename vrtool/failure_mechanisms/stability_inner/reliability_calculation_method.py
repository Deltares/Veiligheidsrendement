from __future__ import annotations

from enum import Enum


class ReliabilityCalculationMethod(Enum):
    """
    Definition of the different types of probability definitions.
    """

    SAFETYFACTOR_RANGE = 0
    BETA_RANGE = 1
    BETA_SINGLE = 2

    @staticmethod
    def is_valid(calculation_method: ReliabilityCalculationMethod) -> None:
        """Validates whether the enum value is valid.

        Args:
            submechanism (PipingFailureSubmechanism): The value to validate.

        Raises:
            ValueError: Raised when submechanism is not a valid value
        """
        if calculation_method not in [
            ReliabilityCalculationMethod.SAFETYFACTOR_RANGE,
            ReliabilityCalculationMethod.BETA_RANGE,
            ReliabilityCalculationMethod.BETA_SINGLE,
        ]:
            raise ValueError(
                f"Unsupported value of {ReliabilityCalculationMethod.__name__}."
            )
