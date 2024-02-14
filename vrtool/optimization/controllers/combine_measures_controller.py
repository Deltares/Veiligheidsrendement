from vrtool.common.enums.combinable_type_enum import CombinableTypeEnum
from vrtool.optimization.measures.combined_measure import CombinedMeasure
from vrtool.optimization.measures.measure_as_input_protocol import (
    MeasureAsInputProtocol,
)
from vrtool.optimization.measures.section_as_input import SectionAsInput
from vrtool.optimization.measures.sg_measure import SgMeasure
from vrtool.optimization.measures.sh_measure import ShMeasure


class CombineMeasuresController:
    def __init__(self, section: SectionAsInput) -> None:
        self._section = section

    @staticmethod
    def _combine_measures(
        measures: list[MeasureAsInputProtocol],
        allowed_measure_combinations: dict[
            CombinableTypeEnum, list[CombinableTypeEnum | None]
        ],
    ) -> list[CombinedMeasure]:
        _combined_measures = []

        def _find_measures(
            combinable_types: list[CombinableTypeEnum | None],
            measures: list[MeasureAsInputProtocol],
        ) -> list[MeasureAsInputProtocol | None]:
            _measures = []
            for _type in combinable_types:
                if _type is None:
                    _measures.append(None)
                else:
                    _measures.extend(
                        filter(lambda x: x.combine_type == _type, measures)
                    )
            return _measures

        def _find_measure_combinations(
            measures: list[MeasureAsInputProtocol],
            combination: tuple[CombinableTypeEnum, list[CombinableTypeEnum | None]],
        ) -> list[tuple[MeasureAsInputProtocol, list[MeasureAsInputProtocol | None]]]:
            _measure_combinations = []
            _primary_measures = _find_measures([combination[0]], measures)
            for _primary_measure in _primary_measures:
                _measure_combinations.append(
                    (
                        _primary_measure,
                        _find_measures(combination[1], measures),
                    )
                )
            return _measure_combinations

        def _create_combined_measures(
            measure_combination: tuple[
                MeasureAsInputProtocol, list[MeasureAsInputProtocol | None]
            ]
        ) -> list[CombinedMeasure]:
            _combined_measures = []
            for _secondary_measure in measure_combination[1]:
                _combined_measures.append(
                    CombinedMeasure.from_input(
                        measure_combination[0],
                        _secondary_measure,
                    )
                )
            return _combined_measures

        # Loop over allowed combinations
        for _combination in allowed_measure_combinations.items():
            _measure_combinations = _find_measure_combinations(measures, _combination)
            for _measure_combination in _measure_combinations:
                _combined_measures.extend(
                    _create_combined_measures(_measure_combination)
                )
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
