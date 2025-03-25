from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from vrtool.common.dike_traject_info import DikeTrajectInfo
from vrtool.common.enums.computation_type_enum import ComputationTypeEnum
from vrtool.common.enums.mechanism_enum import MechanismEnum
from vrtool.flood_defence_system.cross_sectional_requirements import (
    CrossSectionalRequirements,
)
from vrtool.flood_defence_system.section_reliability import SectionReliability


@dataclass(kw_only=True)
class DikeSection:
    """
    Initialize the DikeSection class, as a general class for a dike section that contains all basic information
    """

    crest_height: float = float("nan")
    cover_layer_thickness: float = float("nan")
    mechanism_data: dict[MechanismEnum, list[tuple[str, ComputationTypeEnum]]] = field(
        default_factory=dict
    )
    section_reliability: SectionReliability = field(default_factory=SectionReliability)
    TrajectInfo: DikeTrajectInfo = None
    name: str = ""
    Length: float = float("nan")
    InitialGeometry: pd.DataFrame = None
    houses: pd.DataFrame = None
    with_measures: bool = True
    pleistocene_level: float = float("nan")
    flood_damage: float = float("nan")
    sensitive_fraction_piping: float = 0
    sensitive_fraction_stability_inner: float = 0

    def get_cross_sectional_properties(self) -> CrossSectionalRequirements:
        """
        Gets the cross sectional properties of this dike section required to
        calculate section reliability length effect.

        Returns:
            CrossSectionalRequirements: Dataclass with required parameters for length effect calculations.
        """
        return CrossSectionalRequirements(
            dike_section_length=self.Length,
            dike_traject_b_piping=self.TrajectInfo.bPiping
            if isinstance(self.TrajectInfo, DikeTrajectInfo)
            else 0.0,
            dike_traject_b_stability_inner=self.TrajectInfo.bStabilityInner
            if isinstance(self.TrajectInfo, DikeTrajectInfo)
            else 0.0,
            dike_section_a_piping=self.sensitive_fraction_piping,
            dike_section_a_stability_inner=self.sensitive_fraction_stability_inner,
        )
