from enum import Enum
from re import sub


class MechanismEnum(Enum):
    OVERFLOW = 1
    STABILITY_INNER = 2
    PIPING = 3
    REVETMENT = 4
    HYDRAULIC_STRUCTURES = 5
    INVALID = 99

    def __str__(self) -> str:
        return self.name

    # TODO: delete this method after rationalizing the testdata (VRTOOL-296)
    def get_old_name(self) -> str:
        """Get name according to old naming convention (CamelCase)"""
        return "".join(x.lower().capitalize() or "_" for x in self.name.split("_"))

    @classmethod
    def get_enum(cls, enum_name: str) -> Enum:
        """Return matching enum for name"""

        def _normalize_name(in_name: str) -> str:
            """Convert string to match naming convention (upper snake)"""
            if not in_name:
                return cls.INVALID.name
            return sub(r"(?<!^)(?=[A-Z])", "_", in_name.strip()).upper()

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
