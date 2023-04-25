from vrtool.flood_defence_system.mechanism_reliability_collection import (
    MechanismReliabilityCollection,
)


class FailureMechanismCollection:
    """Class holding a collection of failure mechanisms and their associated information"""

    _failure_mechanisms: dict[str, MechanismReliabilityCollection]

    def __init__(self) -> None:
        self._failure_mechanisms = {}

    def get_available_mechanisms(self) -> set[str]:
        """Gets the available failure mechanisms.

        Returns:
            set[str]: A collection with all the available failure mechanisms.
        """
        return self._failure_mechanisms.keys()

    def get_calculation_years(self) -> list[str]:
        """Gets the collection of years that are calculated.

        Returns:
            list[str]: A collection of the years that are calculated. Empty when the collection contains no failure mechanisms.
        """
        if not self._failure_mechanisms:
            return []

        mechanism_name = list(self._failure_mechanisms)[0]
        return list(
            self.get_mechanism_reliability_collection(mechanism_name).Reliability.keys()
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
        mechanism_name = reliability_collection.mechanism_name
        if mechanism_name in self._failure_mechanisms:
            raise ValueError(f'Mechanism "{mechanism_name}" already added.')

        self._failure_mechanisms[mechanism_name] = reliability_collection

    def get_mechanism_reliability_collection(
        self, mechanism_name: str
    ) -> MechanismReliabilityCollection:
        """Gets the associated collection of reliabilities for the given failure mechanism.

        Args:
            mechanism_name (str): The name of the failure mechanism to retrieve the collection of reliabilities for.

        Returns:
            MechanismReliabilityCollection: A collection of reliabilities. None if the mechanism was not present.
        """
        if mechanism_name not in self._failure_mechanisms:
            return None

        return self._failure_mechanisms[mechanism_name]
