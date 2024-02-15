from typing import Protocol, runtime_checkable

from vrtool.common.enums.combinable_type_enum import CombinableTypeEnum
from vrtool.common.enums.measure_type_enum import MeasureTypeEnum
from vrtool.common.enums.mechanism_enum import MechanismEnum
from vrtool.optimization.measures.mechanism_per_year_probability_collection import (
    MechanismPerYearProbabilityCollection,
)


@runtime_checkable
class MeasureAsInputProtocol(Protocol):
    """stores data for measure in optimization"""

    measure_type: MeasureTypeEnum
    combine_type: CombinableTypeEnum
    cost: float
    year: int
    lcc: float
    mechanism_year_collection: MechanismPerYearProbabilityCollection

    @staticmethod
    def get_concrete_parameters() -> list[str]:
        """
        Gets the concrete parameters of a `MeasureAsInputProtocol` instance that are not defined in the protocol.

        Returns:
            list[str]: List of property names not defined in the `MeasureAsInputProtocol` type.
        """
        pass

    @staticmethod
    def is_mechanism_allowed(mechanism: MechanismEnum) -> bool:
        """
        Check if mechanism is allowed for measure

        Args:
            mechanism (MechanismEnum): Mechanism

        Returns:
            bool: True if allowed
        """
        pass

    @staticmethod
    def get_allowed_mechanisms() -> list[MechanismEnum]:
        """
        Get the allowed mechanisms for the measure

        Returns:
            list[MechanismEnum]: List of Mechanisms
        """
        pass

    @staticmethod
    def get_allowed_measure_combinations() -> (
        dict[CombinableTypeEnum, list[CombinableTypeEnum | None]]
    ):
        """
        Returns the allowed measure type combinations for the measure

        Returns:
            dict[CombinableTypeEnum, list[CombinableTypeEnum | None]]:
                List of measure type combinations (primary: secondary)
                `None` means no secondary measure is needed
        """
        pass
