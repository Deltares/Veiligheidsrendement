import pandas as pd

from vrtool.flood_defence_system.section_reliability import SectionReliability
from vrtool.orm.models.mechanism import Mechanism
from vrtool.orm.models.mechanism_per_section import MechanismPerSection
from vrtool.orm.models.section_data import SectionData


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


def create_mechanism_per_section(section_data: SectionData) -> list[str]:
    def create_combination(mechanism_name: str):
        _mechanism = Mechanism.create(name=mechanism_name)
        MechanismPerSection.create(section=section_data, mechanism=_mechanism)

    _mechanism_names = ["IrrelevantMechanism1", "IrrelevantMechanism2"]
    list(map(create_combination, _mechanism_names))
    return _mechanism_names
