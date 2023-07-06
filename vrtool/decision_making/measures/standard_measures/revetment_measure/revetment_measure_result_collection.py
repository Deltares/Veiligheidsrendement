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
        # We want to output the BETA TARGET and TRANSITION LEVEL
        _output_vector = []
        for _revetment_result in self.revetment_measure_results:
            _output_vector.append([_revetment_result.beta_target, _revetment_result.transition_level])
        return _output_vector

    def get_reliability_values(self) -> list[Any]:
        # We want to output the BETA COMBINED and TOTAL COST
        _output_vector = []
        for _revetment_result in self.revetment_measure_results:
            _output_vector.append([_revetment_result.beta_combined, _revetment_result.cost])
        return _output_vector
