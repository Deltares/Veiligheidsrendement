from vrtool.common.enums.vrtool_enum import VrtoolEnum


class CombinableTypeEnum(VrtoolEnum):
    FULL = 1
    COMBINABLE = 2
    PARTIAL = 3
    REVETMENT = 4
    INVALID = 99

    # TODO: delete this method after rationalizing the testdata (VRTOOL-296)
    def get_old_name(self) -> str:
        """Get name according to old naming convention"""
        return self.name.lower()
