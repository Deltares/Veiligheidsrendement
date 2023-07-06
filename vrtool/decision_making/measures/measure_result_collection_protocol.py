from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class MeasureResultCollectionProtocol(Protocol):
    measure_id: str
    measure_type: str  # TODO: This should be an ENUM
    measure_class: str
    measure_year: int

    def get_measure_input_values(self) -> list[list[Any]]:
        """
        Gets all the possible measure input values that could be (eventually) output to a file or used as dataframe.

        Returns:
            list[list[Any]]: List of lists with all possible relevant values used as measure input.
        """
        pass

    def get_reliability_values(self) -> list[Any]:
        """
        Gets the measure reliability values as a list.
        Note: The return value could be a list containing lists of the aforementioned `measure reliability values`.

        Returns:
            list[Any]: List of lists with all possible values regarding the reliability of this measure.
        """
        pass
