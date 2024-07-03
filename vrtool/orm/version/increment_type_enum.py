from __future__ import annotations

from vrtool.common.enums.vrtool_enum import VrtoolEnum


class IncrementTypeEnum(VrtoolEnum):
    MAJOR = 1
    MINOR = 2
    PATCH = 3
    INVALID = 99

    @classmethod
    def get_enum(cls, enum_name: str) -> IncrementTypeEnum:
        return super().get_enum(enum_name)
