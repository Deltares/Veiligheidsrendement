from numpy import concatenate

from vrtool.decision_making.measures.measure_result_collection_protocol import (
    MeasureResultProtocol,
)
from vrtool.flood_defence_system.section_reliability import SectionReliability


class CustomMeasureResult(MeasureResultProtocol):
    beta_target: float
    section_reliability: SectionReliability
    cost: float
    measure_id: str
    measure_name: str
    measure_year: int
    # TODO: This should be an ENUM
    reinforcement_type: str
    # TODO: This should be an ENUM
    combinable_type: str

    def _get_input_vector(
        self,
        split_params: bool,
    ) -> list:
        if not split_params:
            return [
                self.measure_id,
                self.reinforcement_type,
                self.combinable_type,
                self.measure_year,
                -999,  # yes/no column
                self.cost,
            ]
        return [
            self.measure_id,
            self.reinforcement_type,
            self.combinable_type,
            self.measure_year,
            -999,  # yes/no column
            -999,  # dcrest column
            -999,  # dberm column
            -999,  # beta_target,
            -999,  # transition_level,
            -999,  # l_stab_screen,
            self.cost,
        ]

    def _get_beta_values_for_mechanism(self, mechanism_name: str) -> list[float]:
        if mechanism_name not in self.section_reliability.SectionReliability.index:
            # If a mechanism has not been computed it is irrelevant so the beta is assumed to be 10.
            return [10.0] * len(self.section_reliability.SectionReliability.columns)

        return self.section_reliability.SectionReliability.loc[mechanism_name].values

    def get_measure_output_values(
        self, split_params: bool, beta_columns: list[str]
    ) -> tuple[list, list]:
        _input_measure = self._get_input_vector(split_params)
        if not beta_columns:
            return _input_measure, []

        _output_betas = (
            concatenate(list(map(self._get_beta_values_for_mechanism, beta_columns)))
            .ravel()
            .tolist()
        )
        return _input_measure, _output_betas
