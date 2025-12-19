from collections import defaultdict
from typing import Any

from vrtool.common.enums.mechanism_enum import MechanismEnum
from vrtool.common.hydraulic_loads.load_input import LoadInput
from vrtool.flood_defence_system.cross_sectional_requirements import (
    CrossSectionalRequirements,
)
from vrtool.flood_defence_system.failure_mechanism_collection import (
    FailureMechanismCollection,
)

BETA_THRESHOLD: float = 8.0


# Class describing safety assessments of a section:
class SectionReliability:
    load: LoadInput
    failure_mechanisms: FailureMechanismCollection
    # Result stored during calculate_section_reliability
    section_pf: dict[int, float]
    mechanism_pf: dict[MechanismEnum, dict[int, float]]

    def __init__(self) -> None:
        self.failure_mechanisms = FailureMechanismCollection()
        self.section_pf = {}
        self.mechanism_pf = {}

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SectionReliability):
            return False

        def reliability_dicts_are_equal(
            left_dict: dict[Any, Any], right_dict: dict[Any, Any]
        ) -> bool:
            if left_dict.keys() != right_dict.keys():
                return False
            for _key, _left_value in left_dict.items():
                _right_value = right_dict[_key]
                if isinstance(_left_value, dict):
                    if not reliability_dicts_are_equal(_left_value, _right_value):
                        return False
                    continue
                if abs(_left_value - _right_value) > 1e-6:
                    return False
            return True

        return reliability_dicts_are_equal(
            self.section_pf, other.section_pf
        ) and reliability_dicts_are_equal(self.mechanism_pf, other.mechanism_pf)

    def _get_upscale_cross_sectional_probability(
        self,
        section_length: float,
        mechanism_pf: float,
        mechanism_a: float,
        mechanism_b: float,
    ) -> float:
        # N = a * L_section / b
        _n_value = max(mechanism_a * section_length / mechanism_b, 1)
        return min(1 - (1 - mechanism_pf) ** _n_value, 1.0 / 2)

    def calculate_section_reliability(
        self, cross_sectional_requirements: CrossSectionalRequirements
    ):
        # This routine translates cross-sectional to section reliability indices
        for _mech in self.failure_mechanisms.get_available_mechanisms():
            for _year in self.failure_mechanisms.get_calculation_years():
                _pf = self.failure_mechanisms.get_mechanism_year_reliability(
                    _mech, _year
                )
                if not _pf:
                    raise ValueError(
                        f"Could not retrieve failure probability for mechanism {_mech} in year {_year}."
                    )

                if _mech in [MechanismEnum.OVERFLOW, MechanismEnum.REVETMENT]:
                    self.set_reliability_for_mechanism(_mech, _year, _pf)
                elif _mech in [MechanismEnum.STABILITY_INNER, MechanismEnum.PIPING]:
                    # underneath one can choose whether to upscale within sections or not:
                    _mechanism_a = (
                        cross_sectional_requirements.dike_section_a_piping
                        if _mech is MechanismEnum.PIPING
                        else cross_sectional_requirements.dike_section_a_stability_inner
                    )
                    _mechanism_b = (
                        cross_sectional_requirements.dike_traject_b_piping
                        if _mech is MechanismEnum.PIPING
                        else cross_sectional_requirements.dike_traject_b_stability_inner
                    )
                    self.set_reliability_for_mechanism(
                        _mech,
                        _year,
                        self._get_upscale_cross_sectional_probability(
                            cross_sectional_requirements.dike_section_length,
                            _pf,
                            _mechanism_a,
                            _mechanism_b,
                        ),
                    )

                if _year not in self.section_pf.keys():
                    self.section_pf[_year] = 0.0
                self.section_pf[_year] += self.mechanism_pf[_mech][_year]

    def set_reliability_for_mechanism(
        self, mechanism: MechanismEnum, year: int, pf: float
    ) -> None:
        """Sets the reliability (failure probability) for a given mechanism and year.

        Args:
            mechanism (MechanismEnum): The failure mechanism to set the reliability for.
            year (int): The year to set the reliability for.
            pf (float): The failure probability to set.
        """
        if mechanism not in self.mechanism_pf:
            self.mechanism_pf[mechanism] = {}
        self.mechanism_pf[mechanism][year] = pf

    def set_reliability_for_section(self, year: int, pf: float) -> None:
        """Sets the reliability (failure probability) for the section in a given year.

        Args:
            year (int): The year to set the reliability for.
            pf (float): The failure probability to set.
        """
        self.section_pf[year] = pf

    def get_reliability_for_mechanism(
        self, mechanism: MechanismEnum, year: int
    ) -> float | None:
        return self.mechanism_pf.get(mechanism, {}).get(year, None)
