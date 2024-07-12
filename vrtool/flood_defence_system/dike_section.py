from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from vrtool.common.dike_traject_info import DikeTrajectInfo
from vrtool.common.enums.mechanism_enum import MechanismEnum
from vrtool.flood_defence_system.section_reliability import SectionReliability


@dataclass(kw_only=True)
class DikeSection:
    """
    Initialize the DikeSection class, as a general class for a dike section that contains all basic information
    """

    crest_height: float = float("nan")
    cover_layer_thickness: float = float("nan")
    mechanism_data: dict[MechanismEnum, tuple[str, str]] = field(
        default_factory=lambda: {}
    )
    section_reliability: SectionReliability = field(default_factory=SectionReliability)
    TrajectInfo: DikeTrajectInfo = None
    name: str = ""
    Length: float = float("nan")
    crest_height: float = float("nan")
    InitialGeometry: pd.DataFrame = None
    houses: pd.DataFrame = None
    with_measures: bool = True
    cover_layer_thickness: float = float("nan")
    pleistocene_level: float = float("nan")
    flood_damage: float = float("nan")
