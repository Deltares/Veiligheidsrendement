from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass, field
import numpy as np
from vrtool.optimization.measures.section_as_input import SectionAsInput

from vrtool.optimization.strategy_input.strategy_input_protocol import (
    StrategyInputProtocol,
)


@dataclass
class StrategyInputTargetReliability(StrategyInputProtocol):
    # We assume all the sections belong to the same DikeTraject.
    section_as_input_dict: dict[str, SectionAsInput] = field(
        default_factory=lambda: defaultdict(SectionAsInput)
    )
    design_method: str = ""

    @classmethod
    def from_section_as_input_collection(
        cls, section_as_input_collection: list[SectionAsInput]
    ) -> StrategyInputTargetReliability:
        return cls(
            section_as_input_dict={
                s.section_name: s for s in section_as_input_collection
            }
        )
