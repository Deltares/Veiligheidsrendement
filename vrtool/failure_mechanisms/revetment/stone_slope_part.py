from dataclasses import dataclass
from vrtool.failure_mechanisms.revetment.slope_part import SlopePartProtocol


MIN_BLOCK = 26.0
MAX_BLOCK = 27.9


@dataclass
class StoneSlopePart(SlopePartProtocol):
    def is_valid(self) -> bool:
        return self.top_layer_type >= MIN_BLOCK and self.top_layer_type <= MAX_BLOCK
