from vrtool.common.enums.combinable_type_enum import CombinableTypeEnum
from vrtool.optimization.measures.combined_measure import CombinedMeasure
from vrtool.optimization.measures.measure_as_input_protocol import (
    MeasureAsInputProtocol,
)
from vrtool.optimization.measures.section_as_input import SectionAsInput
from vrtool.optimization.measures.sg_measure import SgMeasure
from vrtool.optimization.measures.sh_measure import ShMeasure


class MeasureCombineController:
    def __init__(self, section: SectionAsInput) -> None:
        self._section = section

    def _combine_measures(
        self,
        measures: list[MeasureAsInputProtocol],
        allowed_measure_combinations: list[
            tuple[CombinableTypeEnum, CombinableTypeEnum | None]
        ],
    ) -> list[CombinedMeasure]:
        _combined_measures = []

        # Loop over allowed combinations
        for _combination in allowed_measure_combinations:
            _primary_measures = filter(
                lambda x: x.combine_type == _combination[0], measures
            )
            for _primary in _primary_measures:
                # Add measure without combination
                if _combination[1] is None:
                    _combined_measures.append(CombinedMeasure(_primary, None))
                    continue
                # Add combination
                _secondary_measures = filter(
                    lambda x: x.combine_type == _combination[1], measures
                )
                for _secondary in _secondary_measures:
                    _combined_measures.append(CombinedMeasure(_primary, _secondary))

        return _combined_measures

    def combine(self) -> list[CombinedMeasure]:
        """
        Combine measures from section.

        Returns:
            list[CombinedMeasure]: List of combined measures.
        """
        _combined_measures = []

        _combined_measures.extend(
            self._combine_measures(
                self._section.sh_measures, ShMeasure.get_allowed_measure_combinations()
            )
        )
        _combined_measures.extend(
            self._combine_measures(
                self._section.sg_measures, SgMeasure.get_allowed_measure_combinations()
            )
        )

        return _combined_measures
