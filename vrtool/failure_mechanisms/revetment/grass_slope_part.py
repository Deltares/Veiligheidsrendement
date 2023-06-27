from vrtool.failure_mechanisms.revetment.relation_revetment_protocol import (
    RelationRevetmentProtocol,
)
from vrtool.failure_mechanisms.revetment.slope_part import SlopePartProtocol
from dataclasses import dataclass, field


GRASS_TYPE = 20.0


@dataclass
class GrassSlopePart(SlopePartProtocol):
    begin_part: float
    end_part: float
    tan_alpha: float
    top_layer_type: float  # note that e.g. 26.1 is a valid type
    top_layer_thickness: float = float("nan")

    slope_part_relations: list[RelationRevetmentProtocol] = field(
        default_factory=lambda: []
    )

    def is_valid(self) -> bool:
        return self.top_layer_type == GRASS_TYPE
