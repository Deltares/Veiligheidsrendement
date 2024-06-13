from typing import Callable, Iterable

import pytest
from peewee import SqliteDatabase

from tests.orm import empty_db_fixture
from vrtool.common.enums import MechanismEnum
from vrtool.common.enums.combinable_type_enum import CombinableTypeEnum
from vrtool.common.enums.measure_type_enum import MeasureTypeEnum
from vrtool.orm.models.combinable_type import CombinableType
from vrtool.orm.models.custom_measure_detail import CustomMeasureDetail
from vrtool.orm.models.dike_traject_info import DikeTrajectInfo
from vrtool.orm.models.measure import Measure
from vrtool.orm.models.measure_per_section import MeasurePerSection
from vrtool.orm.models.measure_type import MeasureType
from vrtool.orm.models.mechanism import Mechanism
from vrtool.orm.models.mechanism_per_section import MechanismPerSection
from vrtool.orm.models.section_data import SectionData
from vrtool.orm.models.standard_measure import StandardMeasure


@pytest.fixture(name="create_valid_measure")
def get_valid_measure_fixture() -> Iterable[
    Callable[[MeasureTypeEnum, CombinableTypeEnum], Measure]
]:
    """
    Creates a basic measure within a Vrtool database context.
    """

    def _create_custom_measure_detail(measure: Measure) -> None:
        _mech_inst = Mechanism.create(name=MechanismEnum.INVALID.name)
        _any_section = SectionData.get()
        _measure_per_section, _ = MeasurePerSection.get_or_create(
            measure=measure, section=_any_section
        )
        _mechanism_per_section, _ = MechanismPerSection.get_or_create(
            mechanism=_mech_inst, section=_measure_per_section.section
        )
        CustomMeasureDetail.create(
            measure=measure,
            mechanism_per_section=_mechanism_per_section,
            cost=1234.56,
            beta=42.24,
            year=2023,
        )

    def create_valid_measure(
        measure_type: MeasureTypeEnum,
        combinable_type: CombinableTypeEnum,
    ) -> Measure:
        _measure_type = MeasureType.create(name=measure_type.legacy_name)
        _combinable_type = CombinableType.create(name=combinable_type.name)
        _measure = Measure.create(
            measure_type=_measure_type,
            combinable_type=_combinable_type,
            name="Test Measure",
            year=2023,
        )
        if measure_type == MeasureTypeEnum.CUSTOM:
            _create_custom_measure_detail(_measure)
        else:
            StandardMeasure.create(
                measure=_measure,
                crest_step=4.2,
                direction="onwards",
                stability_screen=False,
                max_crest_increase=0.1,
                max_outward_reinforcement=2,
                max_inward_reinforcement=3,
                prob_of_solution_failure=0.4,
                failure_probability_with_solution=0.5,
            )
        return _measure

    yield create_valid_measure


@pytest.fixture
def valid_section_data_without_measures(
    empty_db_fixture: SqliteDatabase,
) -> SectionData:
    """
    Fixture to generate a valid (dummy) `SectionData` for the empty
    database `empty_db_fixture`.
    """
    _traject = DikeTrajectInfo.create(traject_name="A traject")
    _section_data = SectionData.create(
        dike_traject=_traject,
        section_name="E4E5",
        meas_start=1.2,
        meas_end=2.3,
        section_length=3.4,
        in_analysis=True,
        crest_height=4.5,
        annual_crest_decline=5.6,
    )
    return _section_data
