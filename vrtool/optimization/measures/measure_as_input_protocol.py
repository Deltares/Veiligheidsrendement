from __future__ import annotations

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

    measure_result_id: int
    measure_type: MeasureTypeEnum
    combine_type: CombinableTypeEnum
    cost: float
    base_cost: float
    discount_rate: float
    year: int
    mechanism_year_collection: MechanismPerYearProbabilityCollection
    l_stab_screen: float

    def is_initial_measure(self) -> bool:
        """
        Verifies whether this measure can be considered as an
        "initial" measure.

        Returns:
            bool: Whether it's a base / initial measure.
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

    @staticmethod
    def is_combinable_type_allowed(combinable_type: CombinableTypeEnum) -> bool:
        """
        Verifies whether the given combinable type can be created as an instance
        of this `MeasureAsInputProtocol` type.

        Args:
            combinable_type (CombinableTypeEnum): Combinable type to be checked.

        Returns:
            bool: This combinable type can be represented as this
            `MeasureAsInputProtocol` instance.
        """
        pass
