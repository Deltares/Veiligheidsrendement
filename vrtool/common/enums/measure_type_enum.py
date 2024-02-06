from vrtool.common.enums.vrtool_enum import VrtoolEnum


class MeasureTypeEnum(VrtoolEnum):
    SOIL_REINFORCEMENT = 1
    SOIL_REINFORCEMENT_WITH_STABILITY_SCREEN = 2
    STABILITY_SCREEN = 3
    VERTICAL_GEOTEXTILE = 4
    DIAPHRAGM_WALL = 5
    REVETMENT = 6
    CUSTOM = 7
    INVALID = 99

    # TODO: delete this method after rationalizing the testdata (VRTOOL-296)
    def get_old_name(self) -> str:
        """Get name according to old naming convention"""
        if self.name.find("REINFORCEMENT") > 0:  # Space separated
            return self.name.lower().replace("_", " ").capitalize()
        else:  # Space Separated
            return self.name.lower().replace("_", " ").title()

    @classmethod
    def get_enum(cls, enum_name: str) -> VrtoolEnum:
        """Return matching enum for name"""

        def _normalize_name(in_name: str) -> str:
            """Convert string to match naming convention (UPPER SNAKE)"""
            if not in_name:
                return cls.INVALID.name
            return in_name.strip().replace(" ", "_").upper()

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
