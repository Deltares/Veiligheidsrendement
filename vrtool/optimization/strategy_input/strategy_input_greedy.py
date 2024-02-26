from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from pandas import DataFrame as df

from vrtool.common.enums.mechanism_enum import MechanismEnum
from vrtool.decision_making.strategy_evaluation import split_options
from vrtool.optimization.measures.combined_measure import CombinedMeasure
from vrtool.optimization.measures.section_as_input import SectionAsInput
from vrtool.optimization.measures.sg_measure import SgMeasure
from vrtool.optimization.strategy_input.strategy_input_protocol import (
    StrategyInputProtocol,
)
from vrtool.probabilistic_tools.combin_functions import CombinFunctions


@dataclass
class StrategyInputGreedy(StrategyInputProtocol):
    design_method: str = ""
    options: dict[str, df] = field(default_factory=dict)
    options_height: list[dict[str, df]] = field(default_factory=list)
    options_geotechnical: list[dict[str, df]] = field(default_factory=list)
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

        def _get_independent_probability_of_failure(
            probability_of_failure_lookup: dict[str, np.array]
        ) -> np.array:
            return CombinFunctions.combine_probabilities(
                probability_of_failure_lookup,
                [m.name for m in SgMeasure.get_allowed_mechanisms()],
            )

        _strategy_input = cls()

        # Define options
        _strategy_input.options = {
            _s.section_name: OldMappingHelper.get_section_options(_s)
            for _s in section_as_input_collection
        }
        _strategy_input.options_height, _strategy_input.options_geotechnical = (
            split_options(
                _strategy_input.options, list(section_as_input_collection[0].mechanisms)
            )
        )

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
        _strategy_input.Pf = OldMappingHelper.get_probabilities(
            section_as_input_collection,
            mechanisms,
            _strategy_input._num_sections,
            _strategy_input._max_sh,
            _strategy_input._max_sg,
            _strategy_input._max_year,
        )
        _strategy_input.LCCOption = OldMappingHelper.get_lifecycle_cost(
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


class OldMappingHelper:

    @staticmethod
    def get_section_options(section: SectionAsInput) -> df:
        _options_dict: dict[tuple, Any] = defaultdict(list)
        _years = [*range(section.min_year, section.max_year)]

        for i, _comb in enumerate(section.aggregated_measure_combinations):

            _options_dict[("ID", "")].append(_comb.sg_combination.combined_id)
            _options_dict[("type", "")].append(
                _comb.sg_combination.combined_measure_type
            )
            _options_dict[("class", "")].append(_comb.sg_combination.measure_class)
            _options_dict[("year", "")].append(_comb.sg_combination.year)
            _options_dict[("yes/no", "")].append(_comb.sg_combination.yesno)
            _options_dict[("dcrest", "")].append(_comb.sh_combination.dcrest)
            _options_dict[("dberm", "")].append(_comb.sg_combination.dberm)
            _options_dict[("transition_level", "")].append(
                _comb.sh_combination.transition_level
            )
            _options_dict[("beta_target", "")].append(_comb.sh_combination.beta_target)
            _options_dict[("cost", "")].append(_comb.lcc)
            _options_dict[("combined_db_index", "")].append(
                _comb.sg_combination.combined_db_index
            )

            # Get betas for all years (Sh of Sg)
            for _mech in section.mechanisms:
                _betas = _comb.sh_combination.mechanism_year_collection.get_betas(
                    _mech, _years
                )
                if len(_betas) == 0:
                    _betas = _comb.sg_combination.mechanism_year_collection.get_betas(
                        _mech, _years
                    )
                for y, _beta in enumerate(_betas):
                    if (_mech.name, _years[y]) not in _options_dict.keys():
                        _options_dict[(_mech.name, _years[y])] = np.zeros(
                            len(section.aggregated_measure_combinations)
                        )
                    _options_dict[(_mech.name, _years[y])][i] = _beta

        # Add section for all years
        for _year in _years:
            _options_dict[("Section", _year)] = np.zeros(
                len(section.aggregated_measure_combinations)
            )

        return df(_options_dict)

    @staticmethod
    def get_probabilities(
        sections: list[SectionAsInput],
        mechanisms: set[MechanismEnum],
        num_sections: int,
        max_sh: int,
        max_sg: int,
        max_year: int,
    ) -> dict[str, np.ndarray]:

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

        _pf: dict[str, np.ndarray] = {}

        for _mech in mechanisms:
            # Initialize datastructure:
            if _mech == MechanismEnum.OVERFLOW:
                _pf[_mech.name] = np.full((num_sections, max_sh + 1, max_year), 1.0)
            elif _mech == MechanismEnum.REVETMENT:
                _pf[_mech.name] = np.full((num_sections, max_sh + 1, max_year), 1.0e-18)
            else:
                _pf[_mech.name] = np.full((num_sections, max_sg + 1, max_year), 1.0)

            # Loop over sections
            for n, _section in enumerate(sections):
                _probs = _get_pf_for_mech(
                    _mech, _section, _pf[_mech.name].shape[1:], max_year
                )
                _pf[_mech.name][n, 0 : len(_probs), :] = _probs

        return _pf

    @staticmethod
    def get_lifecycle_cost(
        sections: list[SectionAsInput], num_sections: int, max_sh: int, max_sg: int
    ) -> np.ndarray:
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
