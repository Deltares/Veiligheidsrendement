from __future__ import annotations

from dataclasses import dataclass

from vrtool.common.enums.mechanism_enum import MechanismEnum

@dataclass
class MechanismPerYear:
    mechanism: MechanismEnum
    year: int
    probability: float
