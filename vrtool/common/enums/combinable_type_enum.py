from __future__ import annotations

from vrtool.common.enums.vrtool_enum import VrtoolEnum


class CombinableTypeEnum(VrtoolEnum):
    FULL = 1
    COMBINABLE = 2
    PARTIAL = 3
    REVETMENT = 4
    INVALID = 99

    # TODO: delete this property after rationalizing the testdata (VRTOOL-296)
    @property
    def legacy_name(self) -> str:
        """Get name according to old naming convention"""
        return self.name.lower()

    @classmethod
    def get_enum(cls, enum_name: str) -> CombinableTypeEnum:
        return super().get_enum(enum_name)
