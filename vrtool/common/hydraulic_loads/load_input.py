from dataclasses import dataclass, field

import openturns as ot


@dataclass
class LoadInput:
    # class to store load data
    distribution: dict[int, ot.Distribution] = field(default_factory=dict)
