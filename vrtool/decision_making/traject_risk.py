import numpy as np

from vrtool.common.enums.mechanism_enum import MechanismEnum
from vrtool.optimization.measures.sg_measure import SgMeasure
from vrtool.optimization.measures.sh_measure import ShMeasure


class TrajectRisk:
    """
    Class to calculate the risk of a traject, depending on the applied measure.
    """

    _probability_of_failure: dict[MechanismEnum, np.ndarray] = {}
    _annual_damage: np.ndarray = np.array([], dtype=float)

    def __init__(self, Pf: dict[MechanismEnum, np.ndarray], D: np.ndarray):
        self._probability_of_failure = Pf
        self._annual_damage = D

    @property
    def mechanisms(self) -> list[MechanismEnum]:
        return list(self._probability_of_failure.keys())

    @property
    def num_sections(self) -> int:
        if not self.mechanisms:
            return 0
        return self._probability_of_failure[self.mechanisms[0]].shape[0]

    @property
    def num_measures(self) -> int:
        if not self.mechanisms:
            return 0
        return max(
            self._probability_of_failure[_mech].shape[1] for _mech in self.mechanisms
        )

    @property
    def num_sh_measures(self) -> int:
        if not self.mechanisms:
            return 0
        return max(
            self._probability_of_failure[_mech].shape[1]
            for _mech in self.mechanisms
            if _mech in ShMeasure.get_allowed_mechanisms()
        )

    @property
    def num_sg_measures(self) -> int:
        if not self.mechanisms:
            return 0
        return max(
            self._probability_of_failure[_mech].shape[1]
            for _mech in self.mechanisms
            if _mech in SgMeasure.get_allowed_mechanisms()
        )

    @property
    def num_years(self) -> int:
        if not self.mechanisms:
            return 0
        return self._probability_of_failure[self.mechanisms[0]].shape[2]

    def get_initial_probabilities_copy(
        self, mechanisms: list[MechanismEnum]
    ) -> dict[str, np.ndarray]:
        """
        Get a copy of the initial probabilities of failure for a list of mechanisms.
        If a mechanism is not present in the traject, the probabilities are set to zero.

        Args:
            mechanisms (list[MechanismEnum]): List of mechanisms to get the initial probabilities for.

        Returns:
            dict[MechanismEnum, np.ndarray]: The initial probabilities of failure for the mechanisms [N, Sh/Sg, t].
        """
        _init_probabilities = {}
        for _mech in mechanisms:
            _init_probabilities[_mech] = np.copy(
                self._get_mechanism_probabilities(_mech)
            )
        return _init_probabilities

    def get_section_probabilities(
        self, section: int, mechanism: MechanismEnum
    ) -> np.ndarray:
        """
        Get the probabilities of failure for a section for a mechanism.

        Args:
            mechanism (MechanismEnum): The mechanism to get the probabilities for.

        Returns:
            np.ndarray: The probabilities of failure for the section [Sh/Sg, t].
                        (zeros if the mechanism is not present in the traject)
        """
        if mechanism not in self._probability_of_failure:
            return np.zeros([self.num_measures, self.num_years])
        return self._probability_of_failure[mechanism][section, :, :]

    def get_measure_probabilities(
        self, measure: tuple[int, int, int], mechanism: MechanismEnum
    ) -> np.ndarray:
        """
        Get the probabilities of failure for a mechanism for a specific measure.

        Args:
            measure (tuple[int, int, int]): The measure to get the probabilities for.
            mechanism (MechanismEnum): The mechanism to get the probabilities for.

        Raises:
            ValueError: The mechanism is not in the allowed mechanisms.

        Returns:
            np.ndarray: The probabilities of failure for the measure [t].
                        (zeros if the mechanism is not present in the traject)
        """
        if mechanism not in self._probability_of_failure:
            return np.zeros([self.num_sections, self.num_years])
        _section = measure[0]
        if mechanism in ShMeasure.get_allowed_mechanisms():
            _measure = measure[1]
        elif mechanism in SgMeasure.get_allowed_mechanisms():
            _measure = measure[2]
        else:
            raise ValueError(f"Mechanism {mechanism} not in allowed mechanisms")
        return self._probability_of_failure[mechanism][_section, _measure, :]

    def get_measure_risk(
        self, measure: tuple[int, int, int], mechanism: MechanismEnum
    ) -> np.ndarray:
        """
        Get the risks for a mechanism for a specific measure.

        Args:
            measure (tuple[int, int, int]): The measure to get the risks for.
            mechanism (MechanismEnum): The mechanism to get the risk for.

        Raises:
            ValueError: The mechanism is not in the allowed mechanisms.

        Returns:
            np.ndarray: The calculated risks for the measure [t].
                        (zeros if the mechanism is not present in the traject)
        """
        return self._annual_damage * self.get_measure_probabilities(measure, mechanism)

    def get_mechanism_risk(self, mechanism: MechanismEnum) -> np.ndarray:
        """
        Get the risks for a specific mechanism for a traject, based on the initial probabilities.

        Args:
            mechanism (MechanismEnum): The mechanism to get the risk for.

        Returns:
            np.ndarray: The calculated risks for the mechanism [N, t].
        """
        return self._annual_damage * self._get_mechanism_probabilities(mechanism)

    def get_independent_risk(self) -> np.ndarray:
        """
        Get the independent risks for a traject, based on the initial probabilities.

        Args:
            mechanism (MechanismEnum): The mechanism to get the risk for.

        Returns:
            np.ndarray: The calculated independent risks [N, t].
        """
        return self._annual_damage * self._get_independent_probabilities()

    def _get_mechanism_probabilities(self, mechanism: MechanismEnum) -> np.ndarray:
        """
        Get the probabilities of failure for a mechanism based on the initial probabilities.

        Args:
            mechanism (MechanismEnum): The mechanism to get the probabilities for.

        Returns:
            np.ndarray: The probabilities of failure for the mechanism [N, t].
                        (zeros if the mechanism is not present in the traject)
        """
        if mechanism not in self._probability_of_failure:
            return np.zeros([self.num_sections, self.num_years])
        return self._probability_of_failure[mechanism][:, 0, :]

    def _get_independent_probabilities(self) -> np.ndarray:  # [N, t]
        return self._combine_probabilities(
            [MechanismEnum.STABILITY_INNER, MechanismEnum.PIPING], None
        )

    def get_total_risk(self) -> float:
        """
        Calculate the total risk for the initial situation.
        The first Sh and Sg measure are used to calculate this.

        Returns:
            float: The total risk for the traject.
        """
        return np.sum(
            self._annual_damage
            * (
                np.max(
                    self._get_mechanism_probabilities(MechanismEnum.OVERFLOW),
                    axis=0,
                )
                + np.max(
                    self._get_mechanism_probabilities(MechanismEnum.REVETMENT),
                    axis=0,
                )
                + np.sum(self._get_independent_probabilities(), axis=0)
            )
        )

    def _get_mechanism_probabilities_for_measure(
        self, mechanism: MechanismEnum, measure: tuple[int, int, int]
    ) -> np.ndarray:
        """
        Get the probabilities of failure for a mechanism after applying a measure.
        Note: the measure is not applied yet, thus the initial probalities remain the same.

        Args:
            mechanism (MechanismEnum): The mechanism to get the probabilities for.
            measure (tuple[int, int, int]): The measure to apply.

        Raises:
            ValueError: The mechanism is not in the allowed mechanisms.

        Returns:
            np.ndarray: The probabilities of failure for the mechanism after applying the measure [N, t].
                        (zeros if the mechanism is not present in the traject)
        """
        if mechanism not in self._probability_of_failure:
            return np.zeros([self.num_sections, self.num_years])
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
                self._probability_of_failure[mechanism][_section, _measure, :],
                self._probability_of_failure[mechanism][_sections[:], 0, :],
            )
        )

    def _get_independent_probabilities_for_measure(
        self, measure: tuple[int, int, int]
    ) -> np.ndarray:
        """
        Get the independent probabilities of failure applying a measure.
        Note: the measure is not applied yet, thus the initial probalities remain the same.

        Args:
            measure (tuple[int, int, int]): The measure to apply.

        Returns:
            np.ndarray: The independent probabilities of failure after applying the measure [N, t].
        """
        return self._combine_probabilities(
            [MechanismEnum.STABILITY_INNER, MechanismEnum.PIPING], measure
        )

    def get_total_risk_for_measure(self, measure: tuple[int, int, int]) -> float:
        """
        Calculate the total risk for a section after applying a measure on that section.

        Args:
            measure (tuple[int, int, int]): The indices of the section, Sh and Sg measure to apply.

        Returns:
            float: The total risk for the section after applying the measure.
        """
        return np.sum(
            self._annual_damage
            * (
                np.max(
                    self._get_mechanism_probabilities_for_measure(
                        MechanismEnum.OVERFLOW, measure
                    ),
                    axis=0,
                )
                + np.max(
                    self._get_mechanism_probabilities_for_measure(
                        MechanismEnum.REVETMENT, measure
                    ),
                    axis=0,
                )
                + np.sum(
                    self._get_independent_probabilities_for_measure(measure), axis=0
                )
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
            np.ndarray: The combined probability of failure for the selected mechanisms [N, t].
        """
        _combined_probabilities = np.ones(self._annual_damage.shape)

        for _mechanism in selection:
            if _mechanism in self._probability_of_failure.keys():
                if measure:
                    _mech_prob = self._get_mechanism_probabilities_for_measure(
                        _mechanism, measure
                    )
                else:
                    _mech_prob = self._get_mechanism_probabilities(_mechanism)
                _combined_probabilities = _combined_probabilities * (1 - _mech_prob)
        return 1 - _combined_probabilities

    def update_probabilities_for_measure(self, measure: tuple[int, int, int]) -> None:
        """
        Update the probabilities of failure for the initial situation after applying a measure
        by copying the measure possibilities to the initial situation.

        Args:
            measure (tuple[int, int, int]): The section, Sh and Sg measure to apply.
        """
        _section = measure[0]
        for _mech in self.mechanisms:
            if _mech in ShMeasure.get_allowed_mechanisms():
                _measure = measure[1]
            elif _mech in SgMeasure.get_allowed_mechanisms():
                _measure = measure[2]
            else:
                return
            self._probability_of_failure[_mech][_section, 0, :] = (
                self._probability_of_failure[_mech][_section, _measure, :]
            )
