from dataclasses import dataclass


@dataclass
class RevetmentMeasureResult:
    year: int
    beta_target: float
    beta_combined: float
    transition_level: float
    cost: float
