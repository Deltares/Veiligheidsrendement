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
    discount_rate: float
    year: int
    mechanism_year_collection: MechanismPerYearProbabilityCollection
    start_cost: float

    @property
    def lcc(self) -> float:
        """Life cycle cost"""
        pass

    def is_initial_cost_measure(self) -> bool:
        """
        Verifies whether the given measure is considered the "initial measure".
        This happens when its year is 0 but most important when its
        dberm or dcrest are 0 / nan (for `ShMeasure` and `SgMeasure` respectively).

        Returns:
            bool: Whether its an initial measure or not.
        """
        pass

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
