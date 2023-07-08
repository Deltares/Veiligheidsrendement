from vrtool.decision_making.measures.measure_result_collection_protocol import (
    MeasureResultCollectionProtocol,
)
from vrtool.decision_making.measures.standard_measures.revetment_measure.revetment_measure_result import (
    RevetmentMeasureResult,
)
from vrtool.decision_making.measures.standard_measures.revetment_measure.revetment_measure_section_reliability import (
    RevetmentMeasureSectionReliability,
)


class RevetmentMeasureResultCollection(MeasureResultCollectionProtocol):
    revetment_measure_results: list[RevetmentMeasureResult]
    beta_target_results: list[RevetmentMeasureSectionReliability]

    def __init__(self) -> None:
        self.revetment_measure_results = []
        self.beta_target_results = []

    def get_measure_output_values(
        self, split_params: bool, beta_columns: list[str]
    ) -> tuple[list, list]:
        return zip(
            *(
                _bt.get_measure_output_values(split_params, beta_columns)
                for _bt in self.beta_target_results
            )
        )
