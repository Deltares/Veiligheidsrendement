from typing import Protocol, runtime_checkable


@runtime_checkable
class FailureMechanismCalculatorProtocol(Protocol):
    def calculate(self, year: int) -> tuple[float, float]:
        """
        Calculates the reliability and probability of failure of a failure mechanism based on its input arguments.

        Args:
            year (int): The year to calculate the reliability and probability of failure for.

        Returns:
            tuple[float, float]: A tuple containing the reliability and probability of failure.
        """
        pass
