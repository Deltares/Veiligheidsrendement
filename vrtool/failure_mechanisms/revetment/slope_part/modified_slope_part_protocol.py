from typing import Protocol, runtime_checkable
from vrtool.failure_mechanisms.revetment.slope_part import SlopePartProtocol


@runtime_checkable
class ModifiedSlopePartProtocol(Protocol):
    modified_slope_part: SlopePartProtocol
    previous_slope_part: SlopePartProtocol

    def is_valid(self) -> bool:
        """
        Validates whether this `SlopePartProtocol` instance fulfill its predefined requirements.

        Returns:
            bool: Wheter it is valid or not.
        """
        pass
