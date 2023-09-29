from vrtool.decision_making.measures.measure_result_collection_protocol import (
    MeasureResultCollectionProtocol,
    MeasureResultProtocol,
)

_supported_parameters = ["dberm", "dcrest"]


def to_measure_result_collection(
    measure_results: list | dict | MeasureResultCollectionProtocol,
) -> MeasureResultCollectionProtocol:
    """
    Gets the correct `MeasureResultCollectionProtocol` instance given a supported
        type of measure_results. When needed, and supported,
        a convertor will be used to retrieve such structure.

    Args:
        measure_results (list | dict | MeasureResultCollectionProtocol):
            Measure containing measure results data as
            a `list`, `dict` or `MeasureResultCollectionProtocol`.

    Raises:
        ValueError: When the provided `measure_results` does not contain a supported
            `measures` property.

    Returns:
        MeasureResultCollectionProtocol: Valid instance of a
            `MeasureResultCollectionProtocol` to export.
    """

    class MeasureDictAsMeasureResult(MeasureResultProtocol):
        def __init__(self, measure_as_dict: dict) -> None:
            self.measure_id = measure_as_dict.pop("ID", "custom-measure-without-id")
            self.cost = measure_as_dict.pop("Cost")
            self.section_reliability = measure_as_dict.pop("Reliability")
            self.parameters = dict(
                [
                    (k, v)
                    for k, v in measure_as_dict.items()
                    if k.lower() in _supported_parameters
                ]
            )

        def get_measure_result_parameters(self) -> list[dict]:
            return self.parameters

    class MeasureDictListAsMeasureResultCollection(MeasureResultCollectionProtocol):
        def __init__(self, measure_dict_list: list[dict]) -> None:
            self.result_collection = list(
                map(MeasureDictAsMeasureResult, measure_dict_list)
            )

    if isinstance(measure_results, MeasureResultCollectionProtocol):
        return measure_results

    if isinstance(measure_results, list):
        # Converts standard measures (except `SoilReinforcementMeasure`).
        return MeasureDictListAsMeasureResultCollection(measure_results)
    elif isinstance(measure_results, dict):
        # Converts `SoilReinforcementMeasure`.
        return MeasureDictListAsMeasureResultCollection([measure_results])
    raise ValueError(f"Unknown measure type: '{type(measure_results).__name__}'.")
