from dataclasses import dataclass
from vrtool.failure_mechanisms.revetment.slope_part import SlopePartProtocol

ASPHALT_TYPE = 5.0


@dataclass
class AsphaltSlopePart(SlopePartProtocol):
    def is_valid(self) -> bool:
        return self.top_layer_type == ASPHALT_TYPE
