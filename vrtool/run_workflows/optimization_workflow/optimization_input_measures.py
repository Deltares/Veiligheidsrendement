from dataclasses import dataclass
from vrtool.defaults.vrtool_config import VrtoolConfig
from vrtool.flood_defence_system.dike_traject import DikeTraject
from vrtool.optimization.measures.section_as_input import SectionAsInput


@dataclass
class OptimizationInputMeasures:
    vr_config: VrtoolConfig
    selected_traject: DikeTraject
    section_input_collection: list[SectionAsInput]

    @property
    def measure_id_year_list(self) -> list[tuple[int, int]]:
        """
        Gets a list of tuples representing a `MeasureResult` id and its `investment_year`.

        Returns:
            list[tuple[int, int]]: List of tuples `list[tuple[id, investment_year]]`.
        """
        return sorted( 
            list(
            set(
                (mr.measure_result_id, mr.year)
                for _section_input in self.section_input_collection
                for mr in _section_input.measures
            )
        )
        )
