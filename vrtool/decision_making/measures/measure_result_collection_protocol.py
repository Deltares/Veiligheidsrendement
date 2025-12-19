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

    def get_measure_result_parameters(self) -> dict:
        """
        Gets all the existing result parameters related to this `MeasureResultProtocol`.

        Returns:
            dict: Dictionary representing parameter name and value.
        """
        pass


@runtime_checkable
class MeasureResultCollectionProtocol(Protocol):
    result_collection: list[MeasureResultProtocol]
