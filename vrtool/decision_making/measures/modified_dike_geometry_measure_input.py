from __future__ import annotations

import pandas as pd
from dataclasses import dataclass


@dataclass
class ModifiedDikeGeometryMeasureInput:
    """Class containing input properties related to the modification of the dike geometry as a measure."""

    modified_geometry: pd.Series
    area_extra: float
    area_excavated: float
    d_house: float
    d_berm: float
    d_crest: float

    @classmethod
    def from_dictionary(cls, dict: dict) -> ModifiedDikeGeometryMeasureInput:
        return cls(**dict)
