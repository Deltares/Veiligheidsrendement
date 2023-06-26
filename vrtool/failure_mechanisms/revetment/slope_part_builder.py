from typing import Type
from vrtool.failure_mechanisms.revetment.asphalt_slope_part import (
    ASPHALT_TYPE,
    AsphaltSlopePart,
)
from vrtool.failure_mechanisms.revetment.slope_part import SlopePartProtocol
from vrtool.failure_mechanisms.revetment.grass_slope_part import (
    GrassSlopePart,
    GRASS_TYPE,
)
from vrtool.failure_mechanisms.revetment.stone_slope_part import (
    StoneSlopePart,
    MAX_BLOCK,
    MIN_BLOCK,
)


class SlopePartBuilder:
    @staticmethod
    def get_slope_part_type(top_layer_type: float) -> Type[SlopePartProtocol]:
        if top_layer_type == GRASS_TYPE:
            return GrassSlopePart
        elif top_layer_type >= MIN_BLOCK and top_layer_type <= MAX_BLOCK:
            return StoneSlopePart
        elif top_layer_type == ASPHALT_TYPE:
            return AsphaltSlopePart

        raise ValueError(
            "No SlopePart type found for top layer type: {}.".format(top_layer_type)
        )
