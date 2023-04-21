from __future__ import annotations

from enum import Enum


class PipingFailureSubmechanism(Enum):
    """
    Definition of the different piping failure submechanisms.
    """

    PIPING = 0
    HEAVE = 1
    UPLIFT = 2

    @staticmethod
    def is_valid(submechanism: PipingFailureSubmechanism) -> None:
        """Validates whether the enum value is valid.

        Args:
            submechanism (PipingFailureSubmechanism): The value to validate.

        Raises:
            ValueError: Raised when submechanism is not a valid value
        """
        if submechanism not in [
            PipingFailureSubmechanism.PIPING,
            PipingFailureSubmechanism.HEAVE,
            PipingFailureSubmechanism.UPLIFT,
        ]:
            raise ValueError(
                f"Unsupported value of {PipingFailureSubmechanism.__name__}."
            )
