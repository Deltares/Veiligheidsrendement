from __future__ import annotations

from enum import Enum
from re import sub


class VrtoolEnum(Enum):
    def __str__(self) -> str:
        return self.name

    # TODO: delete this method after rationalizing the testdata (VRTOOL-296)
    def get_old_name(self) -> str:
        """Get name according to old naming convention (CamelCase)"""
        return self.name.lower().title().replace("_", "")

    @classmethod
    def get_enum(cls, enum_name: str) -> VrtoolEnum:
        """Return matching enum for name"""

        def _normalize_name(in_name: str) -> str:
            """Convert string to match naming convention (UPPER_SNAKE)"""
            if not in_name:
                return cls.INVALID.name
            return sub(
                r"(?<!^)(?=[A-Z])", "_", in_name.strip().replace(" ", "_")
            ).upper()

        try:
            # Default: enum name exists
            return cls[enum_name]
        except KeyError:
            # Fallback:
            # -> enum name needs to be normalized first
            # -> if still no match: INVALID is returned
            return next(
                (
                    _enum
                    for _enum in list(cls)
                    if _normalize_name(enum_name) == _enum.name
                ),
                cls.INVALID,
            )
