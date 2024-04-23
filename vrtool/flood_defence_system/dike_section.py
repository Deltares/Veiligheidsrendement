from __future__ import annotations

from pathlib import Path

import pandas as pd

from vrtool.common.dike_traject_info import DikeTrajectInfo
from vrtool.common.enums.mechanism_enum import MechanismEnum
from vrtool.common.hydraulic_loads.load_input import LoadInput
from vrtool.defaults.vrtool_config import VrtoolConfig
from vrtool.flood_defence_system.mechanism_reliability_collection import (
    MechanismReliabilityCollection,
)
from vrtool.flood_defence_system.section_reliability import SectionReliability


class DikeSection:
    """
    Initialize the DikeSection class, as a general class for a dike section that contains all basic information
    """

    crest_height: float
    mechanism_data: dict[MechanismEnum, tuple[str, str]]
    section_reliability: SectionReliability
    TrajectInfo: DikeTrajectInfo
    name: str
    InitialGeometry: pd.DataFrame
    Length: float
    houses: pd.DataFrame
    with_measures: bool

    def __init__(self) -> None:
        self.mechanism_data = {}
        self.section_reliability = SectionReliability()
        self.TrajectInfo = None
        self.name = ""
        self.Length = float("nan")
        self.crest_height = float("nan")
        self.InitialGeometry = None
        self.houses = None
        self.with_measures = True
