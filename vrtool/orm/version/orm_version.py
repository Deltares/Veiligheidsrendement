from __future__ import annotations

from dataclasses import dataclass

from vrtool.orm import __version__
from vrtool.orm.version.increment_type_enum import IncrementTypeEnum


@dataclass
class OrmVersion:
    major: int
    minor: int
    patch: int

    def __hash__(self) -> int:
        return (100 * self.major) + (10 * self.minor) + self.patch

    def __le__(self, other: OrmVersion) -> bool:
        return self.__hash__() <= other.__hash__()

    def __gt__(self, other: OrmVersion) -> bool:
        return self.__hash__() > other.__hash__()

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"

    @staticmethod
    def get_increment_type(
        from_version: OrmVersion, to_version: OrmVersion
    ) -> IncrementTypeEnum:
        if from_version.major < to_version.major:
            return IncrementTypeEnum.MAJOR
        elif from_version.minor < to_version.minor:
            return IncrementTypeEnum.MINOR
        elif from_version.patch < to_version.patch:
            return IncrementTypeEnum.PATCH
        return IncrementTypeEnum.NONE

    @staticmethod
    def parse_version(version_string: str) -> tuple[int, int, int]:
        return tuple(map(int, version_string.split(".")))

    @classmethod
    def from_orm(cls) -> OrmVersion:
        _major, _minor, _patch = cls.parse_version(__version__)
        return cls(major=_major, minor=_minor, patch=_patch)
