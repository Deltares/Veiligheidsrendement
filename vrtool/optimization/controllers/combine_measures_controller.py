from itertools import combinations

from vrtool.common.enums.combinable_type_enum import CombinableTypeEnum
from vrtool.optimization.measures.combined_measure import CombinedMeasure
from vrtool.optimization.measures.measure_as_input_protocol import (
    MeasureAsInputProtocol,
)
from vrtool.optimization.measures.mechanism_per_year_probability_collection import (
    MechanismPerYearProbabilityCollection,
)
from vrtool.optimization.measures.section_as_input import SectionAsInput
from vrtool.optimization.measures.sg_measure import SgMeasure
from vrtool.optimization.measures.sh_measure import ShMeasure


class CombineMeasuresController:
    def __init__(self, section: SectionAsInput) -> None:
        self._section = section

    @staticmethod
    def combine_measures(
        measures: list[MeasureAsInputProtocol],
        allowed_measure_combinations: dict[
            CombinableTypeEnum, list[CombinableTypeEnum | None]
        ],
        initial_assessment: MechanismPerYearProbabilityCollection,
    ) -> list[CombinedMeasure]:
        """
        Create all possible combinations of measures

        Args:
            measures (list[MeasureAsInputProtocol]): The measures to be combined
            allowed_measure_combinations (dict[ CombinableTypeEnum, list[CombinableTypeEnum  |  None] ]):
                The allowed combinations of measures
            initial_assessment (MechanismPerYearProbabilityCollection): The initial assessment

        Returns:
            list[CombinedMeasure]: The combined measures
        """
        _list_to_combine = measures + [None]
        _prospect_combinations = combinations(_list_to_combine, 2)

        def valid_combination(
            combination: tuple[
                MeasureAsInputProtocol | None, MeasureAsInputProtocol | None
            ]
        ) -> bool:
            _primary, _secondary = combination
            if _primary is None or (
                _primary.combine_type not in allowed_measure_combinations.keys()
            ):
                # Primary measures MIGHT NOT be NONE.
                return False

            _allowed_secondary_combinations = allowed_measure_combinations[
                _primary.combine_type
            ]
            # Secondary measures MIGHT be NONE.
            _combine_type = None
            if isinstance(_secondary, MeasureAsInputProtocol):
                _combine_type = _secondary.combine_type

            return _combine_type in _allowed_secondary_combinations

        _combinations = [
            CombinedMeasure.from_input(_primary, _secondary, initial_assessment, i)
            for i, (_primary, _secondary) in enumerate(
                filter(valid_combination, _prospect_combinations)
            )
        ]

        return _combinations

    def combine(self) -> list[CombinedMeasure]:
        """
        Combine measures from section.

        Returns:
            list[CombinedMeasure]: List of combined measures.
        """
        _combinations = []

        _combinations.extend(
            self.combine_measures(
                self._section.sh_measures,
                ShMeasure.get_allowed_measure_combinations(),
                self._section.initial_assessment,
            )
        )

        _combinations.extend(
            self.combine_measures(
                self._section.sg_measures,
                SgMeasure.get_allowed_measure_combinations(),
                self._section.initial_assessment,
            )
        )

        return _combinations
