from __future__ import annotations

import numpy as np
from dataclasses import dataclass

from vrtool.flood_defence_system.mechanism_input import MechanismInput


@dataclass
class OverflowSimpleInput:
    corrected_crest_height: np.ndarray
    q_crest: np.ndarray
    h_c: np.ndarray
    q_c: np.ndarray
    beta: np.ndarray

    @classmethod
    def from_mechanism_input(
        cls, mechanism_input: MechanismInput, corrected_crest_height: np.ndarray
    ) -> OverflowSimpleInput:
        return cls(
            corrected_crest_height=corrected_crest_height,
            q_crest=mechanism_input.input["q_crest"],
            h_c=mechanism_input.input["h_c"],
            q_c=mechanism_input.input["q_c"],
            beta=mechanism_input.input["beta"],
        )
