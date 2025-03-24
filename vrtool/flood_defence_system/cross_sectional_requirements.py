from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vrtool.common.enums.mechanism_enum import MechanismEnum


@dataclass
class CrossSectionalRequirements:
    cross_sectional_requirement_per_mechanism: dict[MechanismEnum, np.ndarray]

    dike_traject_b_piping: float
    dike_traject_b_stability_inner: float
    dike_section_a_piping: float
    dike_section_a_stability_inner: float
    dike_section_length: float