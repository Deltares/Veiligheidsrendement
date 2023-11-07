from typing import Type
from math import floor
from vrtool.failure_mechanisms.revetment.slope_part.asphalt_slope_part import (
    AsphaltSlopePart,
)
from vrtool.failure_mechanisms.revetment.slope_part.grass_slope_part import (
    GrassSlopePart,
)
from vrtool.failure_mechanisms.revetment.slope_part.slope_part_protocol import (
    SlopePartProtocol,
)
from vrtool.failure_mechanisms.revetment.slope_part.stone_slope_part import (
    StoneSlopePart,
)
MOST_RELEVANT_TOP_LAYER_TYPES = {10:  "Betonblokken met afgeschuinde hoeken of gaten erin",
                                 11:  "Betonblokken",
                                 7:   "Breuksteen, gepenetreerd met asfalt (vol en zat)",
                                 25:  "Breuksteen/teenbestorting",
                                 28:  "Natuursteen",
                                 29:  "Koperslakblokken",
                                 32: "Klinkers, bton of gebakken",
                                 }


class SlopePartBuilder:
    @staticmethod
    def get_slope_part_type(top_layer_type: float) -> Type[SlopePartProtocol]:
        """
        Returns the specific `SlopePartPorotocol` type that matches the given `top_layer_type`.

        Args:
            top_layer_type (float): Value determining its material type.

        Raises:
            ValueError: When the provided `top_layer_type` does not match any expected material type.

        Returns:
            Type[SlopePartProtocol]: Type to use to build a valid instance of a `SlopePartProtocol`.
        """
        if GrassSlopePart.is_grass_part(top_layer_type):
            return GrassSlopePart
        elif StoneSlopePart.is_stone_slope_part(top_layer_type):
            return StoneSlopePart
        elif AsphaltSlopePart.is_asphalt_slope_part(top_layer_type):
            return AsphaltSlopePart
        if floor(top_layer_type) in MOST_RELEVANT_TOP_LAYER_TYPES.keys():
            raise ValueError(
                "No SlopePart type found for top layer type: {} ({}). Ignored in computation.".format(
                    top_layer_type,
                    MOST_RELEVANT_TOP_LAYER_TYPES[round(top_layer_type)]
                )
            )
        else:
            raise ValueError(
                "No SlopePart type found for top layer type: {}. Ignored in computation.".format(top_layer_type)
            )

    @staticmethod
    def build(**kwargs) -> SlopePartProtocol:
        """
        Builds a `SlopePartProtocol` concrete instance based on the provided arguments.

        Returns:
            SlopePartProtocol: Valid object instance of a `SlopePartProtocol`
        """
        _builder_type = SlopePartBuilder.get_slope_part_type(
            kwargs.get("top_layer_type", float("nan"))
        )
        return _builder_type(**kwargs)
