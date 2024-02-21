from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from vrtool.optimization.measures.section_as_input import SectionAsInput

from vrtool.optimization.strategy_input.strategy_input_protocol import (
    StrategyInputProtocol,
)


@dataclass
class StrategyInputTargetReliability(StrategyInputProtocol):
    design_method: str = ""

    options: dict = field(default_factory=dict)

    opt_parameters: dict[str, int] = field(default_factory=dict)
    Pf: dict[str, np.ndarray] = field(default_factory=dict)
    LCCOption: np.ndarray = np.array([])
    D: np.ndarray = np.array([])
    RiskGeotechnical: np.ndarray = np.array([])
    RiskOverflow: np.ndarray = np.array([])
    RiskRevetment: np.ndarray = np.array([])
    _num_sections: int = 0
    _max_year: int = 0
    _max_sg: int = 0
    _max_sh: int = 0

    @classmethod
    def from_section_as_input_collection(
        cls, section_as_input_collection: list[SectionAsInput]
    ) -> StrategyInputTargetReliability:
        return cls()
