from __future__ import annotations

from dataclasses import dataclass

from vrtool.orm import __version__
from vrtool.orm.version.increment_type_enum import IncrementTypeEnum


@dataclass
class OrmVersion:
    """
    Class representing a version of the ORM.
    """

    major: int
    minor: int
    patch: int

    def __eq__(self, other: OrmVersion) -> bool:
        return (
            self.major == other.major
            and self.minor == other.minor
            and self.patch == other.patch
        )

    def __lt__(self, other: OrmVersion) -> bool:
        if self.major < other.major:
            return True
        if self.minor < other.minor:
            return True
        if self.patch < other.patch:
            return True
        return False

    def __le__(self, other: OrmVersion) -> bool:
        if self == other:
            return True
        return self < other

    def __gt__(self, other: OrmVersion) -> bool:
        if self.major > other.major:
            return True
        if self.minor > other.minor:
            return True
        if self.patch > other.patch:
            return True
        return False

    def __ge__(self, other: OrmVersion) -> bool:
        if self == other:
            return True
        return self > other

    def __str__(self) -> str:
        """
        String representation of the version.

        Returns:
            str: String representation of the version in the format "major.minor.patch".
        """
        return f"{self.major}.{self.minor}.{self.patch}"

    def get_increment_type(self, other: OrmVersion) -> IncrementTypeEnum:
        """
        Define the increment type against another version.

        Args:
            other (OrmVersion): Version to compare against.

        Returns:
            IncrementTypeEnum: Type of increment between this and other versions.
        """
        if self.major != other.major:
            return IncrementTypeEnum.MAJOR
        if self.minor != other.minor:
            return IncrementTypeEnum.MINOR
        if self.patch != other.patch:
            return IncrementTypeEnum.PATCH
        return IncrementTypeEnum.NONE

    @classmethod
    def from_orm(cls) -> OrmVersion:
        """
        Create an OrmVersion object from the ORM version file.

        Returns:
            OrmVersion: Object representing the ORM version.
        """

        def parse_version(version_string: str) -> tuple[int, int, int]:
            return tuple(map(int, version_string.split(".")))

        _major, _minor, _patch = parse_version(__version__)
        return cls(major=_major, minor=_minor, patch=_patch)
