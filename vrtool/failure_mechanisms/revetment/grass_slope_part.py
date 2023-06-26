from vrtool.failure_mechanisms.revetment.slope_part import SlopePartProtocol
from dataclasses import dataclass


GRASS_TYPE = 20.0


@dataclass
class GrassSlopePart(SlopePartProtocol):
    def is_valid(self) -> bool:
        return self.top_layer_type == GRASS_TYPE
