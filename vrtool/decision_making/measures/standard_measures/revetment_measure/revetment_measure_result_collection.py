from typing import Any
from vrtool.decision_making.measures.measure_result_collection_protocol import (
    MeasureResultCollectionProtocol,
)
from vrtool.decision_making.measures.standard_measures.revetment_measure.revetment_measure_result import (
    RevetmentMeasureResult,
)


class RevetmentMeasureResultCollection(MeasureResultCollectionProtocol):
    measure_id: str
    measure_name: str
    # TODO: This should be an ENUM
    reinforcement_type: str
    # TODO: This should be an ENUM
    combinable_type: str

    revetment_measure_results: list[RevetmentMeasureResult]

    def get_measure_input_values(self) -> list[list[Any]]:
        return [[]]

    def get_reliability_values(self) -> list[Any]:
        return [[]]
