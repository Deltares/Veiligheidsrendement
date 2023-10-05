from enum import Enum


class MechanismEnum(Enum):
    OVERFLOW = 1
    STABILITY_INNER = 2
    PIPING = 3
    REVETMENT = 4

    @classmethod
    def to_list(cls):  # TODO typehint
        """Return list of active entries"""
        return [_enum for _enum in MechanismEnum if _enum.value > 0]
