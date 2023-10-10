from vrtool.common.enums import MechanismEnum
from vrtool.flood_defence_system.mechanism_reliability_collection import (
    MechanismReliabilityCollection,
)


class FailureMechanismCollection:
    """Class holding a collection of failure mechanisms and their associated information"""

    _failure_mechanisms: dict[MechanismEnum, MechanismReliabilityCollection]

    def __init__(self) -> None:
        self._failure_mechanisms = {}

    def get_available_mechanisms(self) -> set[MechanismEnum]:
        """Gets the available failure mechanisms.

        Returns:
            set[MechanismEnum]: A collection with all the available failure mechanisms.
        """
        return self._failure_mechanisms

    def get_calculation_years(self) -> list[str]:
        """Gets the collection of years that are calculated.

        Returns:
            list[str]: A collection of the years that are calculated. Empty when the collection contains no failure mechanisms.
        """
        if not self._failure_mechanisms:
            return []

        mechanism = list(self._failure_mechanisms)[0]
        return list(
            self.get_mechanism_reliability_collection(mechanism).Reliability.keys()
        )

    def add_failure_mechanism_reliability_collection(
        self,
        reliability_collection: MechanismReliabilityCollection,
    ) -> None:
        """Adds a failure mechanism reliability collection to this collection.

        Args:
            reliability_collection (MechanismReliabilityCollection): The collection of reliabilities for a given failure mechanism.

        Raises:
            ValueError: Raised when a collection with the same failure mechanism was already added.
        """
        mechanism = reliability_collection.mechanism
        if mechanism in self._failure_mechanisms:
            raise ValueError(f'Mechanism "{mechanism}" already added.')

        self._failure_mechanisms[mechanism] = reliability_collection

    def get_mechanism_reliability_collection(
        self, mechanism: MechanismEnum
    ) -> MechanismReliabilityCollection:
        """Gets the associated collection of reliabilities for the given failure mechanism.

        Args:
            mechanism (MechanismEnum): The failure mechanism to retrieve the collection of reliabilities for.

        Returns:
            MechanismReliabilityCollection: A collection of reliabilities. None if the mechanism was not present.
        """
        return self._failure_mechanisms.get(mechanism)

    def get_all_mechanism_reliability_collections(
        self,
    ) -> list[MechanismReliabilityCollection]:
        """Gets all the mechanism reliability collections that are stored within the container.

        Returns:
            list[MechanismReliabilityCollection]: The collection of stored mechanism reliability collections
        """
        return list(self._failure_mechanisms.values())
