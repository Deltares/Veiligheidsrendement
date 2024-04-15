from __future__ import annotations

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

    mechanisms: list[MechanismEnum] = field(default_factory=list)
    mechanism_prob: list[MechanismProbabilities] = field(default_factory=list)
    annual_damage: list[float] = field(default_factory=list)

    @classmethod
    def from_strategy_input(
        cls,
        Pf: dict[str, np.ndarray],
        D: np.ndarray,
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
        _traject_prob.mechanisms = mechanisms
        for _mech in mechanisms:
            _traject_prob.mechanism_prob.append(
                MechanismProbabilities.from_strategy_input(
                    _mech,
                    Pf[_mech.name],
                    sh_idx=sh_idx,
                    sg_idx=sg_idx,
                )
            )
        _traject_prob.annual_damage = D.tolist()
        return _traject_prob

    def get_total_risk(self) -> float:
        """
        Get the total risk for the traject over all mechanisms.

        Returns:
            float: Total risk for the traject.
        """
        return 0
        # np.sum(np.max(overflow_risk, axis=0))
        #         + np.sum(np.max(revetment_risk, axis=0))
        #         + np.sum(independent_risk)

    def combine_probabilities(
        self,
        mechanism_prob: list[MechanismProbabilities],
        selection: list[MechanismEnum],
    ) -> list[float]:
        """
        Calculate the combined probability of failure for a selection of mechanisms.

        Args:
            mechanism_prob (list[MechanismProbabilities]): Probabilities of failure for each mechanism.
            selection (list[MechanismEnum]): Mechanisms to consider.

        Returns:
            list[float]: TODO
        """
        for m, _mechanism in enumerate(
            _mech_prob.mechanism for _mech_prob in mechanism_prob
        ):
            if _mechanism in selection:
                pass
        return 0
