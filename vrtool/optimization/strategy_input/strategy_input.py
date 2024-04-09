from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from pandas import DataFrame as df

from vrtool.common.enums.mechanism_enum import MechanismEnum
from vrtool.optimization.measures.section_as_input import SectionAsInput
from vrtool.optimization.measures.sg_measure import SgMeasure
from vrtool.optimization.strategy_input.legacy_mapping_helper import LegacyMappingHelper
from vrtool.optimization.strategy_input.strategy_input_protocol import (
    StrategyInputProtocol,
)
from vrtool.probabilistic_tools.combin_functions import CombinFunctions


@dataclass
class StrategyInput(StrategyInputProtocol):
    design_method: str = ""
    options: dict[str, df] = field(default_factory=dict)
    options_height: list[dict[str, df]] = field(default_factory=list)
    options_geotechnical: list[dict[str, df]] = field(default_factory=list)
    opt_parameters: dict[str, int] = field(default_factory=dict)
    sections: list[SectionAsInput] = field(default_factory=list)
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
    ) -> StrategyInput:
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

        _strategy_input = cls(sections=section_as_input_collection)

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
        _strategy_input.Pf = LegacyMappingHelper.get_probabilities(
            section_as_input_collection,
            mechanisms,
            _strategy_input._num_sections,
            _strategy_input._max_sh,
            _strategy_input._max_sg,
            _strategy_input._max_year,
        )
        _strategy_input.LCCOption = LegacyMappingHelper.get_lifecycle_cost(
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
