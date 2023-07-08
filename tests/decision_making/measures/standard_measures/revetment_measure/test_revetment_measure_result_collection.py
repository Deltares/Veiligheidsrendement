from vrtool.decision_making.measures.measure_result_collection_protocol import (
    MeasureResultCollectionProtocol,
)
from vrtool.decision_making.measures.standard_measures.revetment_measure.revetment_measure_result_collection import (
    RevetmentMeasureResultCollection,
)


class TestRevetmentMeasureResultCollection:
    def test_init(self):
        _result_collection = RevetmentMeasureResultCollection()
        assert isinstance(_result_collection, RevetmentMeasureResultCollection)
        assert isinstance(_result_collection, MeasureResultCollectionProtocol)
