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

    def _get_input_vector(
        self,
        split_params: bool,
        year: int,
        cost: float,
        beta_target: float,
        transition_level: float,
    ) -> list:
        if not split_params:
            return [
                self.measure_id,
                self.reinforcement_type,
                self.combinable_type,
                year,
                "yes",
                cost,
            ]
        return [
            self.measure_id,
            self.reinforcement_type,
            self.combinable_type,
            year,
            "yes",
            float("nan"),  # dcrest column
            float("nan"),  # dberm column
            beta_target,
            transition_level,
            cost,
        ]

    def get_measure_output_values(self, split_params: bool) -> tuple[list, list]:
        _input_measure = []
        _output_betas = []
        _results_dict = self._get_results_as_dict()
        for _beta_target, _beta_group in _results_dict.items():
            for _transition_level, _transition_group in _beta_group.items():
                _input_measure.append(
                    self._get_input_vector(
                        split_params,
                        _transition_group[0].year,
                        _transition_group[0].cost,
                        _beta_target,
                        _transition_level,
                    )
                )
                _output_betas.append(
                    [
                        _gtl.beta_combined
                        for _gtl in sorted(_transition_group, key=lambda x: x.year)
                    ]
                )
        return _input_measure, _output_betas
