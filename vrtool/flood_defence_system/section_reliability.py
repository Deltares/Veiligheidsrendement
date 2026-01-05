from typing import Any

from vrtool.common.enums.mechanism_enum import MechanismEnum
from vrtool.common.hydraulic_loads.load_input import LoadInput
from vrtool.flood_defence_system.cross_sectional_requirements import (
    CrossSectionalRequirements,
)
from vrtool.flood_defence_system.failure_mechanism_collection import (
    FailureMechanismCollection,
)
from vrtool.probabilistic_tools.probabilistic_functions import beta_to_pf

BETA_THRESHOLD: float = 8.0


class SectionReliability:
    """
    Class describing safety assessments of a section.
    """

    load: LoadInput
    failure_mechanisms: FailureMechanismCollection
    # Result stored during calculate_section_reliability
    _section_pf: dict[int, float]
    _mechanism_pf: dict[MechanismEnum, dict[int, float]]

    def __init__(self) -> None:
        self.failure_mechanisms = FailureMechanismCollection()
        self._section_pf = {}
        self._mechanism_pf = {}

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SectionReliability):
            return False

        def reliability_dicts_are_equal(
            left_dict: dict[Any, Any], right_dict: dict[Any, Any]
        ) -> bool:
            if left_dict.keys() != right_dict.keys():
                return False

            def equal_reliability(key) -> bool:
                _left_value = left_dict[key]
                _right_value = right_dict[key]
                if isinstance(_left_value, dict):
                    return reliability_dicts_are_equal(_left_value, _right_value)
                return abs(_left_value - _right_value) < 1e-6

            return all(list(map(equal_reliability, left_dict.keys())))

        return reliability_dicts_are_equal(
            self._section_pf, other._section_pf
        ) and reliability_dicts_are_equal(self._mechanism_pf, other._mechanism_pf)

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
        """
        Translate cross-sectional to section reliability indices

        Args:
            cross_sectional_requirements (CrossSectionalRequirements): The cross-sectional requirements containing the necessary parameters.

        Raises:
            ValueError: Failure probability could not be retrieved for a mechanism in a given year.
        """
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
                    self.set_reliability_for_mechanism_year(_mech, _year, _pf)
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
                    self.set_reliability_for_mechanism_year(
                        _mech,
                        _year,
                        self._get_upscale_cross_sectional_probability(
                            cross_sectional_requirements.dike_section_length,
                            _pf,
                            _mechanism_a,
                            _mechanism_b,
                        ),
                    )

                if _year not in self._section_pf.keys():
                    self._section_pf[_year] = 0.0
                self._section_pf[_year] += self._mechanism_pf[_mech][_year]

    def set_reliabilities(self, reliabilities: dict[int, float]) -> None:
        """Sets the reliabilities (failure probabilities) for the section.

        Args:
            reliabilities (dict[int, float]): A dictionary of year to failure probability.
        """
        for _year, _pf in reliabilities.items():
            self.set_reliability_for_year(_year, _pf)

    def set_reliability_for_year(self, year: int, pf: float) -> None:
        """Sets the reliability (failure probability) for the section in a given year.

        Args:
            year (int): The year to set the reliability for.
            pf (float): The failure probability to set.
        """
        self._section_pf[year] = max(pf, beta_to_pf(BETA_THRESHOLD))

    def set_reliabilities_for_mechanism(
        self, mechanism: MechanismEnum, reliabilities: dict[int, float]
    ) -> None:
        """Sets the reliabilities (failure probabilities) for a given mechanism.

        Args:
            mechanism (MechanismEnum): The failure mechanism to set the reliabilities for.
            reliabilities (dict[int, float]): A dictionary of year to failure probability.
        """
        for _year, _pf in reliabilities.items():
            self.set_reliability_for_mechanism_year(mechanism, _year, _pf)

    def set_reliability_for_mechanism_year(
        self, mechanism: MechanismEnum, year: int, pf: float
    ) -> None:
        """Sets the reliability (failure probability) for a given mechanism and year.

        Args:
            mechanism (MechanismEnum): The failure mechanism to set the reliability for.
            year (int): The year to set the reliability for.
            pf (float): The failure probability to set.
        """
        if mechanism not in self._mechanism_pf:
            self._mechanism_pf[mechanism] = {}
        self._mechanism_pf[mechanism][year] = max(pf, beta_to_pf(BETA_THRESHOLD))

    def get_reliabilities(self) -> dict[int, float]:
        """
        Gets the reliabilities (failure probabilities) for the section.

        Returns:
            dict[int, float]: Dictionary of year to failure probability.
        """
        return self._section_pf

    def get_reliability_for_year(self, year: int) -> float | None:
        """
        Gets the reliability (failure probability) for the section in a given year.

        Args:
            year (int): The year to get the reliability for.

        Returns:
            float | None: The failure probability for the given year, or None if not present.
        """
        return self.get_reliabilities().get(year, None)

    def get_reliabilities_for_mechanisms(self) -> dict[MechanismEnum, dict[int, float]]:
        """
        Gets the reliabilities (failure probabilities) for all mechanisms.

        Returns:
            dict[MechanismEnum, dict[int, float]]: Dictionary of mechanism to (dictionary of year to failure probability).
        """
        return self._mechanism_pf

    def get_reliabilities_for_mechanism(
        self, mechanism: MechanismEnum
    ) -> dict[int, float]:
        """
        Gets the reliabilities (failure probabilities) for a given mechanism.

        Args:
            mechanism (MechanismEnum): The failure mechanism to get the reliabilities for.

        Returns:
            dict[int, float]: Dictionary of year to failure probability for the given mechanism.
        """
        return self.get_reliabilities_for_mechanisms().get(mechanism, {})

    def get_reliability_for_mechanism_year(
        self, mechanism: MechanismEnum, year: int
    ) -> float | None:
        """
        Gets the reliability (failure probability) for a given mechanism and year.

        Args:
            mechanism (MechanismEnum): The failure mechanism to get the reliability for.
            year (int): The year to get the reliability for.

        Returns:
            float | None: The failure probability for the given mechanism and year, or None if not present.
        """
        return self.get_reliabilities_for_mechanism(mechanism).get(year, None)
