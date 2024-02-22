from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any
import numpy as np
import pandas as pd
from vrtool.common.enums.measure_type_enum import MeasureTypeEnum
from vrtool.optimization.measures.combined_measure import CombinedMeasure
from vrtool.optimization.measures.measure_as_input_protocol import (
    MeasureAsInputProtocol,
)
from vrtool.optimization.measures.section_as_input import SectionAsInput

from vrtool.optimization.strategy_input.strategy_input_protocol import (
    StrategyInputProtocol,
)
from vrtool.probabilistic_tools.probabilistic_functions import pf_to_beta


@dataclass
class StrategyInputTargetReliability(StrategyInputProtocol):
    # We assume all the sections belong to the same DikeTraject.
    section_as_input_dict: dict[str, SectionAsInput] = field(
        default_factory=lambda: defaultdict(SectionAsInput)
    )
    design_method: str = ""

    # To be removed (phased-out) properties
    options: dict = field(default_factory=dict)

    @classmethod
    def from_section_as_input_collection(
        cls, section_as_input_collection: list[SectionAsInput]
    ) -> StrategyInputTargetReliability:

        _options = {
            _s.section_name: OldMappingHelper.get_section_options(_s)
            for _s in section_as_input_collection
        }

        return cls(
            section_as_input_dict={
                s.section_name: s for s in section_as_input_collection
            },
            options=_options,
        )


class OldMappingHelper:

    @staticmethod
    def get_section_options(section: SectionAsInput) -> pd.DataFrame:
        _options_dict: dict[tuple, Any] = {}

        _options_dict[("id", "")] = []
        _options_dict[("type", "")] = []
        _options_dict[("class", "")] = []
        _options_dict[("year", "")] = []
        _options_dict[("yes/no", "")] = []
        _options_dict[("dcrest", "")] = []
        _options_dict[("dberm", "")] = []
        _options_dict[("beta_target", "")] = []
        _options_dict[("transition_level", "")] = []
        _options_dict[("cost", "")] = []
        _options_dict[("combined_db_index", "")] = []
        for i, _comb in enumerate(section.combined_measures):
            _options_dict[("id", "")].append(_comb.combined_id)
            _options_dict[("type", "")].append(_comb.combined_measure_type)
            _options_dict[("class", "")].append(_comb.measure_class)
            _options_dict[("year", "")].append(_comb.year)
            _options_dict[("yes/no", "")].append(_comb.yesno)
            _options_dict[("dcrest", "")].append(_comb.dcrest)
            _options_dict[("dberm", "")].append(_comb.dberm)
            _options_dict[("transition_level", "")].append(_comb.transition_level)
            _options_dict[("beta_target", "")].append(_comb.beta_target)
            _options_dict[("cost", "")].append(_comb.lcc)
            _options_dict[("combined_db_index", "")].append(_comb.combined_db_index)

            for _prob in _comb.mechanism_year_collection.probabilities:
                if (_prob.mechanism.name, _prob.year) not in _options_dict.keys():
                    _options_dict[(_prob.mechanism.name, _prob.year)] = np.zeros(
                        len(section.combined_measures)
                    )
                _options_dict[(_prob.mechanism.name, _prob.year)][i] = pf_to_beta(
                    _prob.probability
                )

        return pd.DataFrame(_options_dict)
