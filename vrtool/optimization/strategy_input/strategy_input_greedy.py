from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from vrtool.common.enums.mechanism_enum import MechanismEnum
from vrtool.optimization.measures.combined_measure import CombinedMeasure
from vrtool.optimization.measures.section_as_input import SectionAsInput
from vrtool.optimization.measures.sg_measure import SgMeasure
from vrtool.optimization.strategy_input.strategy_input_protocol import (
    StrategyInputProtocol,
)
from vrtool.probabilistic_tools.combin_functions import CombinFunctions


@dataclass
class StrategyInputGreedy(StrategyInputProtocol):
    design_method: str
    opt_parameters: dict[str, int] = field(default_factory=dict)
    Pf: dict[str, np.ndarray] = field(default_factory=dict)
    LCCOption: np.ndarray = np.array([])
    D: np.ndarray = np.array([])
    RiskGeotechnical: np.ndarray = np.array([])
    RiskOverflow: np.ndarray = np.array([])
    RiskRevetment: np.ndarray = np.array([])
    _num_sections: int = 0
    _max_year: int = 0
    _max_sg: int = 0
    _max_sh: int = 0

    @classmethod
    def from_section_as_input_collection(
        cls, section_as_input_collection: list[SectionAsInput]
    ) -> StrategyInputGreedy:
        """
        Maps the aggregate combinations of measures to the legacy output (temporarily).
        """

        def _get_pf_for_measures(
            mech: MechanismEnum,
            combinations: list[CombinedMeasure],
            dims: tuple[int, ...],
            max_year: int,
        ) -> np.ndarray:
            _probs = np.zeros(dims)
            # Add other measures
            for m, _meas in enumerate(combinations):
                _probs[m, :] = _meas.mechanism_year_collection.get_probabilities(
                    mech, list(range(max_year))
                )
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
            # Concatenate both probabilities
            return np.concatenate((np.array(_initial_probs)[None, :], _probs), axis=0)

        def _get_probabilities(
            sections: list[SectionAsInput],
            mechanisms: set[MechanismEnum],
            num_sections: int,
            max_sh: int,
            max_sg: int,
            max_year: int,
        ) -> dict[str, np.ndarray]:
            _pf: dict[str, np.ndarray] = {}

            for _mech in mechanisms:
                # Initialize datastructure:
                if _mech == MechanismEnum.OVERFLOW:
                    _pf[_mech.name] = np.full((num_sections, max_sh + 1, max_year), 1.0)
                elif _mech == MechanismEnum.REVETMENT:
                    _pf[_mech.name] = np.full(
                        (num_sections, max_sh + 1, max_year), 1.0e-18
                    )
                else:
                    _pf[_mech.name] = np.full((num_sections, max_sg + 1, max_year), 1.0)

                # Loop over sections
                for n, _section in enumerate(sections):
                    _probs = _get_pf_for_mech(
                        _mech, _section, _pf[_mech.name].shape[1:], max_year
                    )
                    _pf[_mech.name][n, 0 : len(_probs), :] = _probs

            return _pf

        def _get_combination_idx(
            comb: CombinedMeasure, combinations: list[CombinedMeasure]
        ) -> int:
            """
            Find the index of the combination in the list of combinations of measures.

            Args:
                comb (CombinedMeasure): The combination at hand.
                combinations (list[CombinedMeasure]): LIs of all combined measures.

            Returns:
                int: Index of the combined measures in the list.
            """
            return next((i for i, c in enumerate(combinations) if c == comb), -1)

        def _get_lifecycle_cost(
            sections: list[SectionAsInput], num_sections: int, max_sh: int, max_sg: int
        ) -> np.ndarray:
            _lcc: np.ndarray = np.array([])
            _lcc = np.full((num_sections, max_sh + 1, max_sg + 1), 1e99)

            for n, _section in enumerate(sections):
                _lcc[n, 0, 0] = 0.0
                for _aggr in _section.aggregated_measure_combinations:
                    _sh_idx = _get_combination_idx(
                        _aggr.sh_combination, _section.sh_combinations
                    )
                    _sg_idx = _get_combination_idx(
                        _aggr.sg_combination, _section.sg_combinations
                    )
                    _lcc[n, _sh_idx + 1, _sg_idx + 1] = _aggr.lcc
            return _lcc

        def _get_independent_probability_of_failure(
            probability_of_failure_lookup: dict[str, np.array]
        ) -> np.array:
            return CombinFunctions.combine_probabilities(
                probability_of_failure_lookup,
                [m.name for m in SgMeasure.get_allowed_mechanisms()],
            )

        # Initialize StrategyInputGreedy
        _strategy_input = cls()

        # Define general parameters
        _strategy_input._num_sections = len(section_as_input_collection)
        _strategy_input._max_year = max(s.max_year for s in section_as_input_collection)
        _strategy_input._max_sg = max(
            map(len, (s.sg_combinations for s in section_as_input_collection))
        )
        _strategy_input._max_sh = max(
            map(len, (s.sh_combinations for s in section_as_input_collection))
        )
        _strategy_input.opt_parameters = {
            "N": _strategy_input._num_sections,
            "T": _strategy_input._max_year,
            "Sg": _strategy_input._max_sg + 1,
            "Sh": _strategy_input._max_sh + 1,
        }

        # Populate probabilities and lifecycle cost datastructures per section(/mechanism)
        mechanisms = set(
            mech for sect in section_as_input_collection for mech in sect.mechanisms
        )
        _strategy_input.Pf = _get_probabilities(
            section_as_input_collection,
            mechanisms,
            _strategy_input._num_sections,
            _strategy_input._max_sh,
            _strategy_input._max_sg,
            _strategy_input._max_year,
        )
        _strategy_input.LCCOption = _get_lifecycle_cost(
            section_as_input_collection,
            _strategy_input._num_sections,
            _strategy_input._max_sh,
            _strategy_input._max_sg,
        )

        # Decision variables for discounted damage [T,]
        _strategy_input.D = np.array(
            section_as_input_collection[0].flood_damage
            * (
                1
                / (
                    (1 + section_as_input_collection[0].measures[0].discount_rate)
                    ** np.arange(0, _strategy_input._max_year, 1)
                )
            )
        )

        # Calculate expected damage
        # - for overflow//piping/slope stability
        _strategy_input.RiskGeotechnical = _get_independent_probability_of_failure(
            _strategy_input.Pf
        ) * np.tile(
            _strategy_input.D.T,
            (_strategy_input._num_sections, _strategy_input._max_sg + 1, 1),
        )

        _strategy_input.RiskOverflow = _strategy_input.Pf[
            MechanismEnum.OVERFLOW.name
        ] * np.tile(
            _strategy_input.D.T,
            (_strategy_input._num_sections, _strategy_input._max_sh + 1, 1),
        )

        # - for revetment
        if MechanismEnum.REVETMENT in mechanisms:
            _strategy_input.RiskRevetment = _strategy_input.Pf[
                MechanismEnum.REVETMENT.name
            ] * np.tile(
                _strategy_input.D.T,
                (_strategy_input._num_sections, _strategy_input._max_sh + 1, 1),
            )
        else:
            _strategy_input.RiskRevetment = np.zeros(
                (
                    _strategy_input._num_sections,
                    _strategy_input._max_sh + 1,
                    _strategy_input._max_year,
                )
            )

        return _strategy_input

    @property
    def Cint_h(self) -> np.ndarray:
        # Decision variables for executed options [N, Sh]
        return np.zeros((self._num_sections, self._max_sh))

    @property
    def Cint_g(self) -> np.ndarray:
        # Decision variables for executed options [N, Sg]
        return np.zeros((self._num_sections, self._max_sg))

    @property
    def Dint(self) -> np.ndarray:
        # Decision variables for weakest overflow section with dims [N,Sh]
        return np.zeros((self._num_sections, self._max_sh))
