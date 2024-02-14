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

    @staticmethod
    def _combine_measures(
        measures: list[MeasureAsInputProtocol],
        allowed_measure_combinations: dict[
            CombinableTypeEnum, list[CombinableTypeEnum | None]
        ],
    ) -> list[CombinedMeasure]:
        _combined_measures = []

        # Loop over allowed combinations
        for _primary_type in allowed_measure_combinations.keys():
            # Find primary measures
            for _primary_measure in filter(
                lambda x: x.combine_type == _primary_type, measures
            ):
                for _secondary_type in allowed_measure_combinations[_primary_type]:
                    # If no secondary is needed, add primary without secondary measure
                    if _secondary_type is None:
                        _combined_measures.append(
                            CombinedMeasure.from_input(_primary_measure, None)
                        )
                        continue
                    # Add combination of primary and secondary measure
                    for _secondary_measure in filter(
                        lambda x: x.combine_type == _secondary_type, measures
                    ):
                        _combined_measures.append(
                            CombinedMeasure.from_input(
                                _primary_measure, _secondary_measure
                            )
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
