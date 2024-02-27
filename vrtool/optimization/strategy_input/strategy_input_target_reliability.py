from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from vrtool.optimization.measures.section_as_input import SectionAsInput
from vrtool.optimization.strategy_input.strategy_input_protocol import (
    StrategyInputProtocol,
)


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

            # Get betas for all years
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

        return pd.DataFrame(_options_dict)
