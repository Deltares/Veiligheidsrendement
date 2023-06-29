from typing import Type
from vrtool.failure_mechanisms.revetment.slope_part.asphalt_slope_part import (
    ASPHALT_TYPE,
    AsphaltSlopePart,
)
from vrtool.failure_mechanisms.revetment.slope_part.slope_part_protocol import (
    SlopePartProtocol,
)
from vrtool.failure_mechanisms.revetment.slope_part.grass_slope_part import (
    GrassSlopePart,
    GRASS_TYPE,
    GrassSlopePart,
)
from vrtool.failure_mechanisms.revetment.slope_part.stone_slope_part import (
    StoneSlopePart,
    MAX_BLOCK,
    MIN_BLOCK,
    StoneSlopePart,
)


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
        if top_layer_type == GRASS_TYPE:
            return GrassSlopePart
        elif top_layer_type >= MIN_BLOCK and top_layer_type <= MAX_BLOCK:
            return StoneSlopePart
        elif top_layer_type == ASPHALT_TYPE:
            return AsphaltSlopePart

        raise ValueError(
            "No SlopePart type found for top layer type: {}.".format(top_layer_type)
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
