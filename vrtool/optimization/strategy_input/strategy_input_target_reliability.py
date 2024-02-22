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
            _s.section_name: OldMappingHelper.get_options(_s.combined_measures)
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
    def get_combined_measure_id_type(
        measure_id_dict: dict[MeasureTypeEnum, str], combined_measure: CombinedMeasure
    ) -> tuple[int, str]:
        """Get or generate the sequence id for the measure type."""

        _primary_measure = combined_measure.primary
        _secondary_measure = combined_measure.secondary

        def _get_measure_type_id_tuple(
            measure: MeasureAsInputProtocol,
        ) -> tuple[str, str]:
            if measure.measure_type in measure_id_dict.keys():
                return measure_id_dict[measure.measure_type]

            # Create new entry
            _type_primary = measure.measure_type.get_old_name()
            _id_primary = 1
            if measure_id_dict.values():
                _id_primary = max([v[0] for v in measure_id_dict.values()]) + 1

            measure_id_dict[measure.measure_type] = (
                _id_primary,
                _type_primary,
            )
            return _id_primary, _type_primary

        _id_primary, _type_primary = _get_measure_type_id_tuple(_primary_measure)

        if not _secondary_measure:
            return (_id_primary, _type_primary)

        # Combine with secondary.
        _id_secondary, _type_secondary = _get_measure_type_id_tuple(_secondary_measure)
        _id = f"{_id_primary}+{_id_secondary}"
        _type = f"{_type_primary}+{_type_secondary}"

        return (_id, _type)

    @staticmethod
    def _get_yesno(comb: CombinedMeasure) -> int | str:
        # this is for greedy
        if comb.primary.measure_type in [
            MeasureTypeEnum.VERTICAL_GEOTEXTILE,
            MeasureTypeEnum.DIAPHRAGM_WALL,
            MeasureTypeEnum.STABILITY_SCREEN,
        ]:
            return "yes"
        return -999

    @staticmethod
    def _get_db_index(comb: CombinedMeasure) -> list[int]:
        _db_index = [comb.primary.measure_result_id]
        if comb.secondary:
            _db_index.append(comb.secondary.measure_result_id)
        return _db_index

    @staticmethod
    def _get_measure_year(
        primary_measure: MeasureAsInputProtocol,
        secondary_measure: MeasureAsInputProtocol | None,
    ) -> int | list[int]:
        """Get the year of the measure."""
        if not secondary_measure:
            return [primary_measure.year]
        return [primary_measure.year, secondary_measure.year]

    @staticmethod
    def get_options(combined_measures: list[CombinedMeasure]):
        _options_dict: dict[tuple, Any] = {}
        _measure_id_dict: dict[MeasureTypeEnum, tuple[str, str]] = defaultdict(
            lambda: (str, str)
        )
        for _comb in combined_measures:
            _id, _type = OldMappingHelper.get_combined_measure_id_type(
                _measure_id_dict, _comb
            )
            _options_dict[("id", "")] = str(_id)
            _options_dict[("type", "")] = _type
            _options_dict[("class", "")] = _comb.class_name
            _options_dict[("year", "")] = _comb.combined_years
        return pd.DataFrame(_options_dict)
