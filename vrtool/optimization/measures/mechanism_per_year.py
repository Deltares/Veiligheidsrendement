from dataclasses import dataclass

from vrtool.common.enums.mechanism_enum import MechanismEnum
from vrtool.probabilistic_tools.probabilistic_functions import pf_to_beta


@dataclass
class MechanismPerYear:
    mechanism: MechanismEnum
    year: int
    probability: float

    @property
    def beta(self) -> float:
        return pf_to_beta(self.probability)
