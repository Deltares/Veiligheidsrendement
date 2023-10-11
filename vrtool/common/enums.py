from enum import Enum
from re import sub


class VrtoolEnum(Enum):
    def __str__(self) -> str:
        return self.name

    def _denormalize_name(self, in_name: str) -> None | str:
        """Get name according to old naming convention (CamelCase)"""
        return "".join(x.lower().capitalize() or "_" for x in in_name.split("_"))

    # TODO: delete this method after rationalizing the testdata (VRTOOL-296)
    def get_old_name(self) -> str:
        """Get name according to old naming convention (CamelCase)"""
        return self._denormalize_name(self.name)

    @classmethod
    def _normalize_name(cls, in_name: str) -> None | str:
        """Convert string to match naming convention (upper snake)"""
        if not in_name:
            return None
        return sub(r"(?<!^)(?=[A-Z])", "_", in_name).upper()

    @classmethod
    def get_enum(cls, enum_name: str) -> None | Enum:
        """Return matching enum for name"""

        try:
            # Default: enum name exists
            return cls[enum_name]
        except KeyError:
            try:
                # Fallback: enum name needs to be normalized first
                return next(
                    _enum
                    for _enum in list(cls)
                    if cls._normalize_name(enum_name) == _enum.name
                )
            except StopIteration:
                return None


class MeasureTypeEnum(VrtoolEnum):
    SOIL_REINFORCEMENT = 1
    SOIL_REINFORCEMENT_WITH_STABILITY_SCREEN = 2
    STABILITY_SCREEN = 3
    VERTICAL_GEOTEXTILE = 4
    DIAPHRAGM_WALL = 5
    REVETMENT = 6
    CUSTOM = 7
    INVALID = 99


class MechanismEnum(VrtoolEnum):
    OVERFLOW = 1
    STABILITY_INNER = 2
    PIPING = 3
    REVETMENT = 4
