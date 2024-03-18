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

        Returns:
            list[CombinedMeasure]: The combined measures
        """
        _list_to_combine = measures + [None]
        _prospect_combinations = combinations(_list_to_combine, 2)

        def valid_combination(
            primary: MeasureAsInputProtocol | None,
            secondary: MeasureAsInputProtocol | None,
        ) -> bool:
            if primary is None or (
                primary.combine_type not in allowed_measure_combinations.keys()
            ):
                # Primary measures MIGHT NOT be NONE.
                return False

            _allowed_secondary_combinations = allowed_measure_combinations[
                primary.combine_type
            ]
            # Secondary measures MIGHT be NONE.
            _combine_type = None
            if isinstance(secondary, MeasureAsInputProtocol):
                _combine_type = secondary.combine_type

            return _combine_type in _allowed_secondary_combinations

        return [
            CombinedMeasure.from_input(_primary, _secondary, initial_assessment)
            for (_primary, _secondary) in _prospect_combinations
            if valid_combination(_primary, _secondary)
        ]

    def combine(self) -> list[CombinedMeasure]:
        """
        Combine measures from section.

        Returns:
            list[CombinedMeasure]: List of combined measures.
        """
        _sh_combinations = self.combine_measures(
            self._section.sh_measures,
            ShMeasure.get_allowed_measure_combinations(),
            self._section.initial_assessment,
        )
        for i, _comb in enumerate(_sh_combinations):
            _comb.combination_idx = i

        _sg_combinations = self.combine_measures(
            self._section.sg_measures,
            SgMeasure.get_allowed_measure_combinations(),
            self._section.initial_assessment,
        )
        for i, _comb in enumerate(_sg_combinations):
            _comb.combination_idx = i

        return _sh_combinations + _sg_combinations
