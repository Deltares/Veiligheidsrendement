from __future__ import annotations

from vrtool.common.enums.vrtool_enum import VrtoolEnum


class ComputationTypeEnum(VrtoolEnum):
    NONE = 0
    HRING = 1
    SEMIPROB = 2
    SIMPLE = 3
    DSTABILITY = 4
    DIRECTINPUT = 5
    SAFE = 6
    INVALID = 99

    @property
    def legacy_name(self) -> str:
        if self.name == "NONE":
            return ""
        return self.name

    @classmethod
    def get_enum(cls, enum_name: str) -> ComputationTypeEnum:
        return super().get_enum(enum_name)
