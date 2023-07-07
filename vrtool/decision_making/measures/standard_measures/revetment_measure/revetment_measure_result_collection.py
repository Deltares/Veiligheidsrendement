from typing import Any
from vrtool.decision_making.measures.measure_result_collection_protocol import (
    MeasureResultCollectionProtocol,
)
from vrtool.decision_making.measures.standard_measures.revetment_measure.revetment_measure_result import (
    RevetmentMeasureResult,
)
from itertools import groupby

from vrtool.flood_defence_system.section_reliability import SectionReliability


class RevetmentMeasureBetaTargetResults(MeasureResultCollectionProtocol):
    beta_target: float
    transition_level: float
    section_reliability: SectionReliability
    cost: float
    revetment_measure_results: list[RevetmentMeasureResult]
    measure_id: str
    reinforcement_type: str
    combinable_type: str

    def _get_input_vector(
        self,
        split_params: bool,
        year: int,
    ) -> list:
        if not split_params:
            return [
                self.measure_id,
                self.reinforcement_type,
                self.combinable_type,
                year,
                "yes",
                self.cost,
            ]
        return [
            self.measure_id,
            self.reinforcement_type,
            self.combinable_type,
            year,
            "yes",
            -999,  # dcrest column
            -999,  # dberm column
            self.beta_target,
            self.transition_level,
            self.cost,
        ]

    def get_measure_output_values(self, split_params: bool) -> tuple[list, list]:
        _input_measure = self._get_input_vector(
            split_params,
            float("nan"),
        )
        _output_betas = self.section_reliability.SectionReliability
        return _input_measure, _output_betas


class RevetmentMeasureResultCollection(MeasureResultCollectionProtocol):
    measure_id: str
    measure_name: str
    # TODO: This should be an ENUM
    reinforcement_type: str
    # TODO: This should be an ENUM
    combinable_type: str

    revetment_measure_results: list[RevetmentMeasureResult]
    beta_target_results: list[RevetmentMeasureBetaTargetResults]

    def __init__(self) -> None:
        self.revetment_measure_results = []
        self.beta_target_results = []

    def get_measure_output_values(self, split_params: bool) -> tuple[list, list]:
        return zip(
            *(
                _bt.get_measure_output_values(split_params)
                for _bt in self.beta_target_results
            )
        )
