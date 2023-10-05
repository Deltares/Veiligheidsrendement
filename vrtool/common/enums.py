from enum import Enum
from re import sub


def _normalize_name(in_name) -> str:
    """Convert string to match naming convention (upper with _)"""
    return sub(r"(?<!^)(?=[A-Z])", "_", in_name).upper()


class MechanismEnum(Enum):
    OVERFLOW = 1
    STABILITY_INNER = 2
    PIPING = 3
    REVETMENT = 4

    def __str__(self) -> str:
        return self.name

    @classmethod
    def get_enum(cls, enum_name: str) -> None | Enum:
        """Return matching enum for name"""

        if enum_name.isupper():
            try:
                return cls[enum_name]
            except KeyError:
                return None

        try:
            return next(
                _enum for _enum in list(cls) if _normalize_name(enum_name) == _enum.name
            )
        except StopIteration:
            return None
