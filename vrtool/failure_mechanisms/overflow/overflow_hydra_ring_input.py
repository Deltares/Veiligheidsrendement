from __future__ import annotations

import pandas as pd
from dataclasses import dataclass

from vrtool.flood_defence_system.mechanism_input import MechanismInput


@dataclass
class OverflowHydraRingInput:
    h_crest: float
    d_crest: float
    hc_beta: pd.DataFrame

    @classmethod
    def from_mechanism_input(
        cls, mechanism_input: MechanismInput
    ) -> OverflowHydraRingInput:
        return cls(
            h_crest=mechanism_input["h_crest"],
            d_crest=mechanism_input["d_crest"],
            hc_beta=mechanism_input["hc_beta"],
        )
