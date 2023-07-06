from typing import Any
from vrtool.decision_making.measures.measure_result_collection_protocol import (
    MeasureResultCollectionProtocol,
)
from vrtool.decision_making.measures.standard_measures.revetment_measure.revetment_measure_result import (
    RevetmentMeasureResult,
)
from itertools import groupby


class RevetmentMeasureResultCollection(MeasureResultCollectionProtocol):
    measure_id: str
    measure_name: str
    # TODO: This should be an ENUM
    reinforcement_type: str
    # TODO: This should be an ENUM
    combinable_type: str

    revetment_measure_results: list[RevetmentMeasureResult]

    def _get_results_as_dict(self) -> dict:
        def get_sorted(
            measures: list[RevetmentMeasureResult], lambda_expression
        ) -> dict:
            _sorted_list = sorted(measures, key=lambda_expression)
            return groupby(_sorted_list, key=lambda_expression)

        _results_dict = {}
        for _beta_key, _beta_group in get_sorted(
            self.revetment_measure_results, lambda x: x.beta_target
        ):
            _results_dict[_beta_key] = {}
            for _transition_key, _transition_group in get_sorted(
                _beta_group, lambda x: x.transition_level
            ):
                _results_dict[_beta_key][_transition_key] = list(_transition_group)
        return _results_dict

    def get_measure_input_values(self, split_params: bool) -> list[list[Any]]:
        # We want to output the BETA TARGET, TRANSITION LEVEL and TOTAL COST
        _output_vector = []
        _results_dict = self._get_results_as_dict()
        for _beta_group in _results_dict.values():
            for _transition_group in _beta_group.values():
                for _revetment_result in _transition_group:
                    _input_vector = [
                        self.measure_id,
                        self.reinforcement_type,
                        self.combinable_type,
                        _revetment_result.year,
                        "yes",
                        _revetment_result.cost,
                    ]
                    if split_params:
                        # dcrest column
                        _input_vector.insert(5, float("nan"))
                        # dberm column
                        _input_vector.insert(5, float("nan"))
                        _input_vector.insert(5, _revetment_result.beta_target)
                        _input_vector.insert(5, _revetment_result.transition_level)
                    _output_vector.append(_input_vector)
        return _output_vector

    def get_reliability_values(self) -> list[Any]:
        # We want to output ONLY the BETA COMBINED (in correct order) for all the available years (from `VrtoolConfig.T`).
        _output_vector = []
        _results_dict = self._get_results_as_dict()
        for _group_by_beta_target in _results_dict.values():
            for _group_by_transition_level in _group_by_beta_target.values():
                _output_vector.append(
                    [_gtl.beta_combined for _gtl in _group_by_transition_level]
                )
        return _output_vector
