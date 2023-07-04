from dataclasses import dataclass, field

from vrtool.failure_mechanisms.revetment.relation_revetment_protocol import (
    RelationRevetmentProtocol,
)
from vrtool.failure_mechanisms.revetment.slope_part.slope_part_protocol import (
    SlopePartProtocol,
)

ASPHALT_TYPE = 5.0


@dataclass
class AsphaltSlopePart(SlopePartProtocol):
    begin_part: float
    end_part: float
    tan_alpha: float
    top_layer_type: float  # note that e.g. 26.1 is a valid type
    top_layer_thickness: float = float("nan")

    slope_part_relations: list[RelationRevetmentProtocol] = field(
        default_factory=lambda: []
    )

    def is_valid(self) -> bool:
        return self.is_asphalt_slope_part(self.top_layer_type)

    @staticmethod
    def is_asphalt_slope_part(top_layer_type: float) -> bool:
        return top_layer_type == ASPHALT_TYPE
