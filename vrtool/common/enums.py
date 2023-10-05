from enum import Enum
from re import sub


class MechanismEnum(Enum):
    OVERFLOW = 1
    STABILITY_INNER = 2
    PIPING = 3
    REVETMENT = 4

    @classmethod
    def get_enum(cls, enum_name: str) -> None | Enum:
        """Return matching enum for name"""

        def _convert_name(in_name) -> str:
            """Convert string to match naming convention (upper with _)"""
            return sub(r"(?<!^)(?=[A-Z])", "_", in_name).upper()

        try:
            return next(
                _enum for _enum in list(cls) if _convert_name(enum_name) == _enum.name
            )
        except StopIteration:
            return None
