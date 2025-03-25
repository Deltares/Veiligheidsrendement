from __future__ import annotations

from vrtool.common.enums.vrtool_enum import VrtoolEnum


class StepTypeEnum(VrtoolEnum):
    UNKNOWN = 0
    SINGLE = 1
    BUNDLING = 2

    @classmethod
    def get_enum(cls, enum_name: str) -> StepTypeEnum:
        return super().get_enum(enum_name)
