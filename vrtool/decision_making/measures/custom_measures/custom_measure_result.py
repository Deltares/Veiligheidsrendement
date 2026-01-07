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
