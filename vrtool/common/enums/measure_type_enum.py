from vrtool.common.enums.vrtool_enum import VrtoolEnum


class MeasureTypeEnum(VrtoolEnum):
    SOIL_REINFORCEMENT = 1
    SOIL_REINFORCEMENT_WITH_STABILITY_SCREEN = 2
    STABILITY_SCREEN = 3
    VERTICAL_PIPING_SOLUTION = 4
    DIAPHRAGM_WALL = 5
    ANCHORED_SHEETPILE = 6
    REVETMENT = 7
    CUSTOM = 8
    INVALID = 99

    # TODO: delete this property after rationalizing the testdata (VRTOOL-296)
    @property
    def legacy_name(self) -> str:
        """Get name according to old naming convention"""
        if self.name.find("REINFORCEMENT") > 0:
            # Capitalize first char of string
            return self.name.lower().replace("_", " ").capitalize()
        else:
            # Capitalize first char of each word
            return self.name.lower().replace("_", " ").title()
