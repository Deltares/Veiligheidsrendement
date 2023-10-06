import pandas as pd
import pytest

from tests import test_data
from vrtool.common.enums import MechanismEnum
from vrtool.flood_defence_system.section_reliability import SectionReliability
from vrtool.orm.models.mechanism import Mechanism
from vrtool.orm.models.mechanism_per_section import MechanismPerSection
from vrtool.orm.models.section_data import SectionData


@pytest.fixture
def section_reliability_with_values() -> SectionReliability:
    _section_reliability = SectionReliability()
    _reliability_file = test_data.joinpath(
        "section_reliability_export", "reliability_results.csv"
    )
    assert _reliability_file.exists()

    _reliability_df = pd.read_csv(_reliability_file, index_col=0)
    _section_reliability.SectionReliability = _reliability_df

    yield _section_reliability


def create_required_mechanism_per_section(
    section_data: SectionData, mechanism_available_list: list[MechanismEnum]
) -> None:
    _added_mechanisms = []
    for mechanism in mechanism_available_list:
        _mechanism, _ = Mechanism.get_or_create(name=mechanism.name)
        _added_mechanisms.append(
            MechanismPerSection.create(section=section_data, mechanism=_mechanism.name)
        )
