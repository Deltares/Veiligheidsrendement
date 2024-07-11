from __future__ import annotations

from vrtool.common.enums.vrtool_enum import VrtoolEnum


class IncrementTypeEnum(VrtoolEnum):
    NONE = 0
    MAJOR = 1
    MINOR = 2
    PATCH = 3
    INVALID = 99

    @staticmethod
    def get_supported_increments() -> list[IncrementTypeEnum]:
        """
        Returns the list of supported increments.

        Returns:
            list[IncrementTypeEnum]: Increments supported for mgiration.
        """
        return [IncrementTypeEnum.PATCH, IncrementTypeEnum.NONE]

    def is_supported(self) -> bool:
        """
        Verifies whether this type of increment is supported for migration.

        Returns:
            bool: Increment can be migrated.
        """
        return self in self.get_supported_increments()
