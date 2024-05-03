from vrtool.decision_making.measures.custom_measures.custom_measure_result import (
    CustomMeasureResult,
)
from vrtool.decision_making.measures.measure_result_collection_protocol import (
    MeasureResultCollectionProtocol,
)


class CustomMeasureResultCollection(MeasureResultCollectionProtocol):
    result_collection: list[CustomMeasureResult]
