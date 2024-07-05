import logging

import numpy as np

from vrtool.common.enums.mechanism_enum import MechanismEnum
from vrtool.optimization.measures.combined_measures.combined_measure_base import (
    CombinedMeasureBase,
)
from vrtool.optimization.measures.section_as_input import SectionAsInput


class LegacyMappingHelper:
    @staticmethod
    def get_probabilities(
        sections: list[SectionAsInput],
        mechanisms: set[MechanismEnum],
        num_sections: int,
        max_sh: int,
        max_sg: int,
        max_year: int,
    ) -> dict[MechanismEnum, np.ndarray]:
        """
        Get the probabilities of failure for all measure combinations for all sections.

        Args:
            sections (list[SectionAsInput]): List of sections data.
            mechanisms (set[MechanismEnum]): Set of mechanisms.
            num_sections (int): Number of sections.
            max_sh (int): Maximum number of Sh combinations of all sections.
            max_sg (int): Maximum number of Sg combinations of all sections.
            max_year (int): Maximum year of all sections.

        Returns:
            dict[MechanismEnum, np.ndarray]: Probabilities of failure (N, Sh, Sg, year)
                per mechanism for each aggregated Sh/Sg measure combination.
        """

        def _get_pf_for_measures(
            mech: MechanismEnum,
            combinations: list[CombinedMeasureBase],
            dims: tuple[int, ...],
            max_year: int,
        ) -> np.ndarray:
            _probs = np.zeros(dims)
            # Add other measures
            for m, _meas in enumerate(combinations):
                _probabilities = _meas.mechanism_year_collection.get_probabilities(
                    mech, list(range(max_year))
                )
                if not _probabilities:
                    return np.array([])
                _probs[m, :] = _probabilities
            return _probs

        def _get_pf_for_mech(
            mech: MechanismEnum,
            section: SectionAsInput,
            dims: tuple[int, ...],
            max_year: int,
        ) -> np.ndarray:
            # Get initial assessment as first measure
            _initial_probs = section.initial_assessment.get_probabilities(
                mech, list(range(max_year))
            )
            # Get probabilities for all measures
            if not any(section.measures):
                return np.array(_initial_probs)[None, :]
            if section.sg_measures[0].is_mechanism_allowed(mech):
                _probs = _get_pf_for_measures(
                    mech, section.sg_combinations, (dims[0] - 1, dims[1]), max_year
                )
            elif section.sh_measures[0].is_mechanism_allowed(mech):
                _probs = _get_pf_for_measures(
                    mech, section.sh_combinations, (dims[0] - 1, dims[1]), max_year
                )
            else:
                raise ValueError("Mechanism not allowed")
            if _probs.size == 0:
                return np.array(_initial_probs)[None, :]
            # Concatenate both probabilities
            return np.concatenate((np.array(_initial_probs)[None, :], _probs), axis=0)

        _pf: dict[MechanismEnum, np.ndarray] = {}

        for _mech in mechanisms:
            # Initialize datastructure:
            if _mech == MechanismEnum.OVERFLOW:
                _pf[_mech] = np.full((num_sections, max_sh + 1, max_year), 1.0)
            elif _mech == MechanismEnum.REVETMENT:
                _pf[_mech] = np.full((num_sections, max_sh + 1, max_year), 1.0e-18)
            else:
                _pf[_mech] = np.full((num_sections, max_sg + 1, max_year), 1.0e-18)

            # Loop over sections
            for n, _section in enumerate(sections):
                try:
                    _probs = _get_pf_for_mech(
                        _mech, _section, _pf[_mech].shape[1:], max_year
                    )
                    if _probs.size == 0:
                        continue
                    _pf[_mech][n, 0 : len(_probs), :] = _probs
                except KeyError:
                    logging.warning(
                        "Mechanism %s not present for section %s.",
                        _mech.name,
                        _section.section_name,
                    )

        return _pf

    @staticmethod
    def get_lifecycle_cost(
        sections: list[SectionAsInput], num_sections: int, max_sh: int, max_sg: int
    ) -> np.ndarray:
        """
        Get the lifecycle costs for all measure combinations for all sections.

        Args:
            sections (list[SectionAsInput]): List of sections data.
            num_sections (int): Number of sections.
            max_sh (int): Maximum number of Sh combinations of all sections.
            max_sg (int): Maximum number of Sg combinations of all sections.

        Returns:
            np.ndarray: Lifecycle cost per section for each aggregated Sh/Sg measure combination (N, Sh, Sg).
        """
        _lcc: np.ndarray = np.array([])
        _lcc = np.full((num_sections, max_sh + 1, max_sg + 1), 1e99)

        for n, _section in enumerate(sections):
            _lcc[n, 0, 0] = 0.0
            for _aggr in _section.aggregated_measure_combinations:
                _lcc[
                    n,
                    _aggr.sh_combination.sequence_nr + 1,
                    _aggr.sg_combination.sequence_nr + 1,
                ] = _aggr.lcc
        return _lcc
