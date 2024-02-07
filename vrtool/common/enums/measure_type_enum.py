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
        if self.name.find("REINFORCEMENT") > 0:
            # Capitalize first char of string
            return self.name.lower().replace("_", " ").capitalize()
        else:
            # Capitalize first char of each word
            return self.name.lower().replace("_", " ").title()
