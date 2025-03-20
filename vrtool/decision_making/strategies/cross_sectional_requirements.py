from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vrtool.common.enums.mechanism_enum import MechanismEnum
from vrtool.flood_defence_system.dike_traject import DikeTraject
from vrtool.optimization.measures.section_as_input import SectionAsInput


@dataclass
class CrossSectionalRequirements:
    cross_sectional_requirement_per_mechanism: dict[MechanismEnum, np.ndarray]

    dike_traject_b_piping: float
    dike_traject_b_stability_inner: float
    dike_section_a_piping: float
    dike_section_a_stability_inner: float
    dike_section_length: float

    @classmethod
    def from_dike_traject_and_section_as_input(cls, dike_traject: DikeTraject, section_as_input: SectionAsInput) -> CrossSectionalRequirements:
        """
        Class method to create a CrossSectionalRequirements object from a DikeTraject object.
        This method calculates the cross-sectional requirements for the dike traject based on the OI2014 approach.
        The cross-sectional requirements are calculated for each mechanism and stored in a dictionary with the mechanism as key and the cross-sectional requirements as value.

        Args:
            dike_traject (DikeTraject): The DikeTraject object for which the cross-sectional requirements are to be calculated.
            section_as_input (SectionAsInput): The section with the specific requirements to be applied for cross sectional computations.

        Returns:
            CrossSectionalRequirements: The CrossSectionalRequirements object with the cross-sectional requirements for the dike traject.
        """
        # compute cross sectional requirements
        n_piping = 1 + (
            dike_traject.general_info.aPiping
            * dike_traject.general_info.TrajectLength
            / dike_traject.general_info.bPiping
        )
        n_stab = 1 + (
            dike_traject.general_info.aStabilityInner
            * dike_traject.general_info.TrajectLength
            / dike_traject.general_info.bStabilityInner
        )
        n_overflow = 1
        n_revetment = 3
        omegaRevetment = 0.1

        _pf_cs_piping = (
            dike_traject.general_info.Pmax
            * dike_traject.general_info.omegaPiping
            / n_piping
        )
        _pf_cs_revetment = dike_traject.general_info.Pmax * omegaRevetment / n_revetment
        _pf_cs_stabinner = (
            dike_traject.general_info.Pmax
            * dike_traject.general_info.omegaStabilityInner
            / n_stab
        )
        _pf_cs_overflow = (
            dike_traject.general_info.Pmax
            * dike_traject.general_info.omegaOverflow
            / n_overflow
        )
        return cls(
            cross_sectional_requirement_per_mechanism={
                MechanismEnum.PIPING: _pf_cs_piping,
                MechanismEnum.STABILITY_INNER: _pf_cs_stabinner,
                MechanismEnum.OVERFLOW: _pf_cs_overflow,
                MechanismEnum.REVETMENT: _pf_cs_revetment,
            },
            dike_traject_b_piping=dike_traject.general_info.bPiping,
            dike_traject_b_stability_inner=dike_traject.general_info.bStabilityInner,
            dike_section_a_piping = section_as_input.a_section_piping,
            dike_section_a_stability_inner = section_as_input.a_section_stability_inner,
            dike_section_length = section_as_input.section_length
        )