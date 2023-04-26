from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class ModifiedDikeGeometryMeasureInput:
    """Class containing input properties related to the modification of the dike geometry as a measure."""

    modified_geometry: pd.Series
    area_extra: float
    area_excavated: float
    d_house: float
    d_berm: float
    d_crest: float
