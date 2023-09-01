import pandas as pd
from vrtool.flood_defence_system.section_reliability import SectionReliability


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
