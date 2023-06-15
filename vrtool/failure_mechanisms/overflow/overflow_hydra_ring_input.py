from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from vrtool.failure_mechanisms.mechanism_input import MechanismInput


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
            h_crest=mechanism_input.input.get("h_crest", float("nan")),
            d_crest=mechanism_input.input.get("d_crest", float("nan")),
            hc_beta=mechanism_input.input["hc_beta"],
        )
