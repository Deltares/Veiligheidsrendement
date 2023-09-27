from vrtool.decision_making.measures.measure_result_collection_protocol import (
    MeasureResultCollectionProtocol,
    MeasureResultProtocol,
)

_supported_parameters = ["dcrest", "dberm"]


class MeasureDictAsMeasureResult(MeasureResultProtocol):
    parameters: dict

    def __init__(self, measure_as_dict: dict) -> None:
        self.cost = measure_as_dict["Cost"]
        self.section_reliability = measure_as_dict["Reliability"]
        self.parameters = dict(
            (x.upper(), measure_as_dict[x])
            for x in measure_as_dict
            if x in _supported_parameters
        )


class MeasureDictListAsMeasureResultCollection(MeasureResultCollectionProtocol):
    def __init__(self, measure_dict_list: list[dict]) -> None:
        self.result_collection = list(
            map(MeasureDictAsMeasureResult, measure_dict_list)
        )
