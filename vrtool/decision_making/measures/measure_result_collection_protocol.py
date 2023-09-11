from typing import Protocol, runtime_checkable
from vrtool.flood_defence_system.section_reliability import SectionReliability


@runtime_checkable
class MeasureResultProtocol(Protocol):
    measure_id: str
    measure_name: str
    section_reliability: SectionReliability
    cost: float
    # TODO: This should be an ENUM
    reinforcement_type: str
    # TODO: This should be an ENUM
    combinable_type: str

    def get_measure_output_values(
        self, split_params: bool, beta_columns: list[str]
    ) -> tuple[list, list]:
        """
        Gets all the possible measure input values and their related betas that could be (eventually) output to a file or used as dataframe.

        Args:
            split_params (bool): Input determining whether extra parameters will be set in the columns.
            beta_columns (list[str]): List of columns for the beta values to be mapped to.

        Returns:
            tuple[list, list]: Tuple containing both the measure input values and the measure reliability (betas) values.
        """
        pass


@runtime_checkable
class MeasureResultCollectionProtocol(Protocol):
    result_collection: list[MeasureResultProtocol]

    def get_measure_output_values(
        self, split_params: bool, beta_columns: list[str]
    ) -> tuple[list, list]:
        """
        Gets all the possible measure input values and their related betas that could be (eventually) output to a file or used as dataframe.

        Args:
            split_params (bool): Input determining whether extra parameters will be set in the columns.
            beta_columns (list[str]): List of columns for the beta values to be mapped to.

        Returns:
            tuple[list, list]: Tuple containing both the measure input values and the measure reliability (betas) values.
        """
        pass
