from __future__ import annotations

from vrtool.common.enums.vrtool_enum import VrtoolEnum


class IncrementTypeEnum(VrtoolEnum):
    NONE = 0
    MAJOR = 1
    MINOR = 2
    PATCH = 3
    INVALID = 99
