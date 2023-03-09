from __future__ import annotations

import numpy as np
from dataclasses import dataclass

from vrtool.failure_mechanisms.mechanism_input import MechanismInput


@dataclass
class OverflowSimpleInput:
    dhc_t: np.ndarray
    h_crest: np.ndarray
    q_crest: np.ndarray
    h_c: np.ndarray
    q_c: np.ndarray
    beta: np.ndarray

    @classmethod
    def from_mechanism_input(
        cls, mechanism_input: MechanismInput
    ) -> OverflowSimpleInput:
        return cls(
            dhc_t=mechanism_input.input["dhc(t)"],
            h_crest=mechanism_input.input["h_crest"],
            q_crest=mechanism_input.input["q_crest"],
            h_c=mechanism_input.input["h_c"],
            q_c=mechanism_input.input["q_c"],
            beta=mechanism_input.input["beta"],
        )
