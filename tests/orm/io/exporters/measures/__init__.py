import pandas as pd
from tests.orm import get_basic_measure_per_section
from vrtool.flood_defence_system.section_reliability import SectionReliability
from vrtool.orm.models.measure_per_section import MeasurePerSection


def create_section_reliability(years: list[int]) -> SectionReliability:
    _section_reliability = SectionReliability()

    _section_reliability.SectionReliability = pd.DataFrame.from_dict(
        {
            "IrrelevantMechanism1": [year / 12.0 for year in years],
            "IrrelevantMechanism2": [year / 13.0 for year in years],
            "Section": [year / 10.0 for year in years],
        },
        orient="index",
        columns=years,
    )
    return _section_reliability

class MeasureResultTestInputData:
    t_columns: list[int]
    expected_cost: float
    section_reliability: SectionReliability
    measure_per_section: MeasurePerSection

    def __init__(self) -> None:
        self.t_columns = [0, 2, 4, 24, 42]
        self.expected_cost = 42.24
        self.section_reliability = create_section_reliability(self.t_columns)
        self.measure_per_section = get_basic_measure_per_section()
