from typing import Callable, Iterator

import pandas as pd
import pytest

from tests import test_data
from vrtool.common.enums.mechanism_enum import MechanismEnum
from vrtool.flood_defence_system.section_reliability import SectionReliability
from vrtool.orm.models.mechanism import Mechanism
from vrtool.orm.models.mechanism_per_section import MechanismPerSection
from vrtool.orm.models.section_data import SectionData


@pytest.fixture(name="create_required_mechanism_per_section")
def create_required_mechanism_per_section_fixture() -> Iterator[
    Callable[[SectionData, list[MechanismEnum]], None]
]:
    """
    Yields a generator of many `MechanismPerSection`.
    """

    def create_required_mechanism_per_section(
        section_data: SectionData, mechanism_available_list: list[MechanismEnum]
    ) -> None:
        _added_mechanisms = []
        for mechanism in mechanism_available_list:
            _mech_inst, _ = Mechanism.get_or_create(name=mechanism.name)
            _added_mechanisms.append(
                MechanismPerSection.create(section=section_data, mechanism=_mech_inst)
            )

    yield create_required_mechanism_per_section


@pytest.fixture(name="section_reliability_with_values")
def get_section_reliability_with_values() -> Iterator[SectionReliability]:
    """
    Yields:
        Iterator[SectionReliability]: Testable `SectionReliability` with verified values.
    """
    _section_reliability = SectionReliability()
    _reliability_file = test_data.joinpath(
        "section_reliability_export", "reliability_results.csv"
    )
    assert _reliability_file.exists()

    _reliability_df = pd.read_csv(_reliability_file, index_col=0)
    _section_reliability.SectionReliability = _reliability_df

    yield _section_reliability
