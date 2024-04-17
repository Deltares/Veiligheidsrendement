from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from vrtool.common.enums.mechanism_enum import MechanismEnum
from vrtool.decision_making.strategy_output.mechanism_probabilities import (
    MechanismProbabilities,
)


@dataclass
class TrajectProbabilities:
    """
    Class to store the probabilities of a trajectory.
    """

    mechanism_probabilities: list[MechanismProbabilities] = field(default_factory=list)
    annual_damage: np.ndarray = np.array([])

    @classmethod
    def from_strategy_input(
        cls,
        prob_failure: dict[str, np.ndarray],
        damage: np.ndarray,
        mechanisms: list[MechanismEnum],
        sh_idx: int,
        sg_idx: int,
    ) -> TrajectProbabilities:
        """
        Create a TrajectProbabilities object from a strategy input for a specific measure.

        Args:
            Pf (dict[str, np.ndarray]): Probabilities of failure for each mechanism.
            D (np.ndarray): Damage costs for each year.
            mechanisms (list[MechanismEnum]): Mechanisms to consider.
            sh_idx (int): Index of the Sh-measure in the strategy input.
            sg_idx (int): Index of the Sg-measure in the strategy input.

        Returns:
            TrajectProbabilities: Probabilities for all mechanisms for all sections of a dike traject.
        """
        _traject_prob = cls()
        for _mech in mechanisms:
            if not _mech.name in prob_failure:
                logging.warning(f"Mechanism {_mech.name} not in prob_failure")
                continue
            _traject_prob.mechanism_probabilities.append(
                MechanismProbabilities.from_strategy_input(
                    _mech,
                    prob_failure[_mech.name],
                    sh_idx=sh_idx,
                    sg_idx=sg_idx,
                )
            )
        _traject_prob.annual_damage = damage
        return _traject_prob

    @property
    def mechanisms(self) -> list[MechanismEnum]:
        return [mech.mechanism for mech in self.mechanism_probabilities]

    @property
    def overflow_probabilities(self) -> np.ndarray:
        return next(
            (
                _mech.get_probabilities()
                for _mech in self.mechanism_probabilities
                if _mech.mechanism == MechanismEnum.OVERFLOW
            ),
            np.zeros_like(self.annual_damage),
        )

    @property
    def revetment_probabilities(self) -> np.ndarray:
        return next(
            (
                _mech.get_probabilities()
                for _mech in self.mechanism_probabilities
                if _mech.mechanism == MechanismEnum.REVETMENT
            ),
            np.zeros_like(self.annual_damage),
        )

    @property
    def independent_probabilities(self) -> np.ndarray:
        return self.combine_probabilities(
            [MechanismEnum.STABILITY_INNER, MechanismEnum.PIPING]
        )

    @property
    def total_risk(self) -> float:
        return np.sum(
            self.annual_damage
            * (
                self.overflow_probabilities
                + self.revetment_probabilities
                + self.independent_probabilities
            )
        )

    def combine_probabilities(
        self,
        selection: list[MechanismEnum],
    ) -> np.ndarray:
        """
        Calculate the combined probability of failure for a selection of mechanisms.

        Args:
            selection (list[MechanismEnum]): Mechanisms to consider.

        Returns:
            np.ndarray: The combined probability of failure for the selected mechanisms.
        """
        _combined_probabilities = np.ones_like(self.annual_damage)
        for _mechanism_probabilities in self.mechanism_probabilities:
            if _mechanism_probabilities.mechanism in selection:
                _combined_probabilities = np.multiply(
                    _combined_probabilities,
                    1 - _mechanism_probabilities.get_probabilities(),
                )
        return 1 - _combined_probabilities
