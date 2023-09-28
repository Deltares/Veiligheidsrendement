from vrtool.decision_making.measures.measure_protocol import MeasureProtocol
from vrtool.decision_making.measures.measure_result_collection_protocol import (
    MeasureResultCollectionProtocol,
    MeasureResultProtocol,
)


class MeasureDictAsMeasureResult(MeasureResultProtocol):
    def __init__(self, measure_as_dict: dict) -> None:
        self.measure_id = measure_as_dict.pop("ID", "custom-measure-without-id")
        self.cost = measure_as_dict.pop("Cost")
        self.section_reliability = measure_as_dict.pop("Reliability")
        self.parameters = measure_as_dict

    def get_measure_result_parameters(self) -> list[dict]:
        return self.parameters


class MeasureDictListAsMeasureResultCollection(MeasureResultCollectionProtocol):
    def __init__(self, measure_dict_list: list[dict]) -> None:
        self.result_collection = list(
            map(MeasureDictAsMeasureResult, measure_dict_list)
        )


def convert_to_measure_result_collection(
    measure: MeasureProtocol,
) -> MeasureResultCollectionProtocol:
    """
    Gets the correct `MeasureResultCollectionProtocol` instance given a valid
        `measure` (`MeasureProtocol`). If needed, and supported,
        a convertor will be used to retrieve such structure.

    Args:
        measure (MeasureProtocol): Measure containing measure results data as
            a `list`, `dict` or `MeasureResultCollectionProtocol`.

    Raises:
        ValueError: When the provided `measure` does not contain a supported
            `measures` property.

    Returns:
        MeasureResultCollectionProtocol: Valid instance of a
            `MeasureResultCollectionProtocol` to export.
    """
    if isinstance(measure.measures, MeasureResultCollectionProtocol):
        return measure.measures

    if isinstance(measure.measures, list):
        return MeasureDictListAsMeasureResultCollection(measure.measures)
    elif isinstance(measure.measures, dict):
        return MeasureDictListAsMeasureResultCollection([measure.measures])
    raise ValueError(f"Unknown measure type: {type(measure).__name__}")
