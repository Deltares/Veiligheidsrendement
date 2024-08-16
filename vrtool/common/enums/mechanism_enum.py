from __future__ import annotations

from vrtool.common.enums.vrtool_enum import VrtoolEnum


class MechanismEnum(VrtoolEnum):
    OVERFLOW = 1
    STABILITY_INNER = 2
    PIPING = 3
    REVETMENT = 4
    HYDRAULIC_STRUCTURES = 5
    INVALID = 99

    @classmethod
    def get_enum(cls, enum_name: str) -> MechanismEnum:
        return super().get_enum(enum_name)
