from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from pandas import DataFrame as df

from vrtool.common.enums.measure_type_enum import MeasureTypeEnum
from vrtool.common.enums.mechanism_enum import MechanismEnum
from vrtool.optimization.measures.combined_measure import CombinedMeasure
from vrtool.optimization.measures.measure_as_input_protocol import (
    MeasureAsInputProtocol,
)
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
        cls, section_measures_input: list[SectionAsInput]
    ) -> StrategyInputGreedy:
        """
        Maps the aggregate combinations of measures to the legacy output (temporarily).
        """

        def _get_section_options(section: SectionAsInput) -> df:
            _measure_id_dict: dict[MeasureTypeEnum, tuple[str, str]] = {}

            def _get_measure_id_type(
                primary_measure: MeasureAsInputProtocol,
                secondary_measure: MeasureAsInputProtocol | None,
            ) -> tuple[str, str]:
                """Get or generate the sequence id for the measure type."""
                if primary_measure.measure_type not in _measure_id_dict.keys():
                    # Create new entry
                    _type_primary = primary_measure.measure_type.get_old_name()
                    if _measure_id_dict.values():
                        _id_primary = str(
                            max([int(v[0]) for v in _measure_id_dict.values()]) + 1
                        )
                    else:
                        _id_primary = "1"
                    _measure_id_dict[primary_measure.measure_type] = (
                        _id_primary,
                        _type_primary,
                    )
                else:
                    _id_primary, _type_primary = _measure_id_dict[
                        primary_measure.measure_type
                    ]
                if secondary_measure:
                    _id_secondary, _type_secondary = _get_measure_id_type(
                        secondary_measure, None
                    )
                    _id = f"{_id_primary}+{_id_secondary}"
                    _type = f"{_type_primary}+{_type_secondary}"
                else:
                    _id = _id_primary
                    _type = _type_primary
                return (_id, _type)

            def _get_measure_class(
                primary_measure: MeasureAsInputProtocol,
                secondary_measure: MeasureAsInputProtocol | None,
            ) -> str:
                """Get the class of the measure."""
                _class = primary_measure.combine_type.get_old_name()
                if secondary_measure:
                    _class += "+" + secondary_measure.combine_type.get_old_name()
                return _class

            def _get_measure_year(
                primary_measure: MeasureAsInputProtocol,
                secondary_measure: MeasureAsInputProtocol | None,
            ) -> int | list[int]:
                """Get the year of the measure."""
                _year = primary_measure.year
                if secondary_measure:
                    _year = [_year, secondary_measure.year]
                return _year

            _options_dict: dict[tuple, Any] = {}
            for _comb in section.combined_measures:
                if not _comb.secondary:
                    _id, _type = _get_measure_id_type(_comb.primary, None)
                else:
                    _id, _type = _get_measure_id_type(_comb.primary, _comb.secondary)
                _options_dict[("id", "")] = _id
                _options_dict[("type", "")] = _type
                _options_dict[("class", "")] = _get_measure_class(
                    _comb.primary, _comb.secondary
                )
                _options_dict[("year", "")] = _get_measure_year(
                    _comb.primary, _comb.secondary
                )
            return df(_options_dict)

        def _get_options(
            section_measures_input: list[SectionAsInput],
        ) -> dict[str, df]:
            options: dict[str, df] = {}
            for _section in section_measures_input:
                options[_section.section_name] = _get_section_options(_section)
            return options

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

        _strategy_input.options = _get_options(section_measures_input)

        # Define general parameters
        _strategy_input._num_sections = len(section_measures_input)
        _strategy_input._max_year = max(s.max_year for s in section_measures_input)
        _strategy_input._max_sg = max(
            map(len, (s.sg_combinations for s in section_measures_input))
        )
        _strategy_input._max_sh = max(
            map(len, (s.sh_combinations for s in section_measures_input))
        )
        _strategy_input.opt_parameters = {
            "N": _strategy_input._num_sections,
            "T": _strategy_input._max_year,
            "Sg": _strategy_input._max_sg + 1,
            "Sh": _strategy_input._max_sh + 1,
        }

        # Populate probabilities and lifecycle cost datastructures per section(/mechanism)
        mechanisms = set(
            mech for sect in section_measures_input for mech in sect.mechanisms
        )
        _strategy_input.Pf = _get_probabilities(
            section_measures_input,
            mechanisms,
            _strategy_input._num_sections,
            _strategy_input._max_sh,
            _strategy_input._max_sg,
            _strategy_input._max_year,
        )
        _strategy_input.LCCOption = _get_lifecycle_cost(
            section_measures_input,
            _strategy_input._num_sections,
            _strategy_input._max_sh,
            _strategy_input._max_sg,
        )

        # Decision variables for discounted damage [T,]
        _strategy_input.D = np.array(
            section_measures_input[0].flood_damage
            * (
                1
                / (
                    (1 + section_measures_input[0].measures[0].discount_rate)
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
