from vrtool.decision_making.measures.custom_measures.custom_measure_result import (
    CustomMeasureResult,
)
from vrtool.decision_making.measures.measure_result_collection_protocol import (
    MeasureResultCollectionProtocol,
)


class CustomMeasureResultCollection(MeasureResultCollectionProtocol):
    result_collection: list[CustomMeasureResult]

    def __init__(self) -> None:
        self.result_collection = []

    def get_measure_output_values(
        self, split_params: bool, beta_columns: list[str]
    ) -> tuple[list, list]:
        if not self.result_collection:
            return ([], [])
        return tuple(
            zip(
                *(
                    _bt.get_measure_output_values(split_params, beta_columns)
                    for _bt in self.result_collection
                )
            )
        )
