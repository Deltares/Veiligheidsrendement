import pandas as pd
import pytest
from tests import test_data
from vrtool.flood_defence_system.section_reliability import SectionReliability


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
