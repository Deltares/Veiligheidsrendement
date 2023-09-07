from numpy import concatenate

from vrtool.decision_making.measures.measure_result_collection_protocol import (
    MeasureResultCollectionProtocol,
)
from vrtool.decision_making.measures.standard_measures.revetment_measure.revetment_measure_result import (
    RevetmentMeasureResult,
)
from vrtool.flood_defence_system.section_reliability import SectionReliability


class RevetmentMeasureSectionReliability(MeasureResultCollectionProtocol):
    beta_target: float
    transition_level: float
    section_reliability: SectionReliability
    cost: float
    revetment_measure_results: list[RevetmentMeasureResult]
    measure_id: str
    measure_name: str
    measure_year: int
    # TODO: This should be an ENUM
    reinforcement_type: str
    # TODO: This should be an ENUM
    combinable_type: str

    def _get_input_vector(
        self,
        split_params: bool,
    ) -> list:
        if not split_params:
            return [
                self.measure_id,
                self.reinforcement_type,
                self.combinable_type,
                self.measure_year,
                -999,  # yes/no column
                self.cost,
            ]
        return [
            self.measure_id,
            self.reinforcement_type,
            self.combinable_type,
            self.measure_year,
            -999,  # yes/no column
            -999,  # dcrest column
            -999,  # dberm column
            self.beta_target,
            self.transition_level,
            self.cost,
        ]

    def get_measure_output_values(
        self, split_params: bool, beta_columns: list[str]
    ) -> tuple[list, list]:
        _input_measure = self._get_input_vector(split_params)
        _output_betas = (
            concatenate(
                [
                    self.section_reliability.SectionReliability.loc[beta_column].values
                    for beta_column in beta_columns
                ]
            )
            .ravel()
            .tolist()
        )
        return _input_measure, _output_betas
