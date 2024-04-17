import numpy as np

from vrtool.common.enums.mechanism_enum import MechanismEnum
from vrtool.optimization.measures.sg_measure import SgMeasure
from vrtool.optimization.measures.sh_measure import ShMeasure


class TrajectRisk:
    """
    Class to calculate the risk of a traject, depending on the applied measure.
    """

    probability_of_failure: dict[MechanismEnum, np.ndarray] = {}
    annual_damage: np.ndarray = np.array([], dtype=float)

    def __init__(self, Pf: dict[str, np.ndarray], D: np.ndarray):
        self.probability_of_failure = {
            MechanismEnum.get_enum(_mech): np.array(_mech_probs, dtype=float)
            for _mech, _mech_probs in Pf.items()
        }
        self.annual_damage = D

    @property
    def mechanisms(self) -> list[MechanismEnum]:
        return list(self.probability_of_failure.keys())

    @property
    def num_sections(self) -> int:
        return self.probability_of_failure[self.mechanisms[0]].shape[0]

    @property
    def num_years(self) -> int:
        return self.probability_of_failure[self.mechanisms[0]].shape[2]

    def get_initial_probabilities(
        self, mechanisms: list[MechanismEnum]
    ) -> dict[MechanismEnum, np.ndarray]:
        """
        Get the initial probabilities of failure for a list of mechanisms.
        If a mechanism is not present in the traject, the probabilities are set to zero.

        Args:
            mechanisms (list[MechanismEnum]): List of mechanisms to get the initial probabilities for.

        Returns:
            dict[MechanismEnum, np.ndarray]: The initial probabilities of failure for the mechanisms.
        """
        _init_probabilities = {}
        for _mech in mechanisms:
            if _mech not in self.probability_of_failure.keys():
                _init_probabilities[_mech] = np.zeros(
                    [self.num_sections, self.num_years]
                )
                continue
            _init_probabilities[_mech] = self.probability_of_failure[_mech][:, 0, :]
        return _init_probabilities

    def get_mechanism_risk(
        self, mechanism: MechanismEnum, measure: tuple[int, int, int] | None
    ) -> np.ndarray:
        return self.annual_damage * self._get_mechanism_probabilities(
            mechanism, measure
        )

    def get_independent_risk(self, measure: tuple[int, int, int] | None) -> np.ndarray:
        return self.annual_damage * self._get_independent_probabilities(measure)

    def _get_mechanism_probabilities(
        self, mechanism: MechanismEnum, measure: tuple[int, int, int] | None
    ) -> np.ndarray:
        if mechanism not in self.probability_of_failure:
            return np.zeros([self.num_sections, self.num_years])
        if not measure:
            return self.probability_of_failure[mechanism][:, 0, :]
        _sections = list(range(self.num_sections))
        _section = measure[0]
        if mechanism in ShMeasure.get_allowed_mechanisms():
            _measure = measure[1]
        elif mechanism in SgMeasure.get_allowed_mechanisms():
            _measure = measure[2]
        else:
            raise ValueError(f"Mechanism {mechanism} not in allowed mechanisms")
        _sections.remove(measure[0])
        return np.row_stack(
            (
                self.probability_of_failure[mechanism][_section, _measure, :],
                self.probability_of_failure[mechanism][_sections[:], 0, :],
            )
        )

    def _get_independent_probabilities(
        self, measure: tuple[int, int, int] | None
    ) -> np.ndarray:
        return self._combine_probabilities(
            [MechanismEnum.STABILITY_INNER, MechanismEnum.PIPING], measure
        )

    def get_total_risk(self) -> float:
        """
        Calculate the total risk for the initial situation.
        The first Sh and Sg measure are used to calculate this.

        Returns:
            float: The total risk for the traject.
        """
        return np.sum(
            self.annual_damage
            * (
                np.max(
                    self._get_mechanism_probabilities(MechanismEnum.OVERFLOW, None),
                    axis=0,
                )
                + np.max(
                    self._get_mechanism_probabilities(MechanismEnum.REVETMENT, None),
                    axis=0,
                )
                + np.sum(self._get_independent_probabilities(None), axis=0)
            )
        )

    def get_total_risk_for_measure(self, measure: tuple[int, int, int] | None) -> float:
        """
        Calculate the total risk for a section after applying a measure on that section.

        Args:
            measure (tuple[int, int, int]): The section, Sh and Sg measure to apply.

        Returns:
            float: The total risk for the section after applying the measure.
        """
        return np.sum(
            self.annual_damage
            * (
                np.max(
                    self._get_mechanism_probabilities(MechanismEnum.OVERFLOW, measure),
                    axis=0,
                )
                + np.max(
                    self._get_mechanism_probabilities(MechanismEnum.REVETMENT, measure),
                    axis=0,
                )
                + np.sum(self._get_independent_probabilities(measure), axis=0)
            )
        )

    def _combine_probabilities(
        self, selection: list[MechanismEnum], measure: tuple[int, int, int] | None
    ) -> np.ndarray:
        """
        Calculate the combined probability of failure for a selection of mechanisms.

        Args:
            selection (list[MechanismEnum]): Mechanisms to consider.

        Returns:
            np.ndarray: The combined probability of failure for the selected mechanisms.
        """
        _combined_probabilities = np.ones(self.annual_damage.shape)

        for _mechanism in selection:
            if _mechanism in self.probability_of_failure.keys():
                _combined_probabilities = _combined_probabilities * (
                    1 - self._get_mechanism_probabilities(_mechanism, measure)
                )
        return 1 - _combined_probabilities
