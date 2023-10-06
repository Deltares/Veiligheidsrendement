import pytest
from peewee import SqliteDatabase

from tests import test_data
from vrtool.orm.models.combinable_type import CombinableType
from vrtool.orm.models.computation_scenario import ComputationScenario
from vrtool.orm.models.computation_type import ComputationType
from vrtool.orm.models.dike_traject_info import DikeTrajectInfo
from vrtool.orm.models.measure import Measure
from vrtool.orm.models.measure_per_section import MeasurePerSection
from vrtool.orm.models.measure_result import MeasureResult
from vrtool.orm.models.measure_type import MeasureType
from vrtool.orm.models.mechanism import Mechanism
from vrtool.orm.models.mechanism_per_section import MechanismPerSection
from vrtool.orm.models.section_data import SectionData
from vrtool.orm.orm_controllers import open_database


@pytest.fixture(autouse=False)
def empty_db_fixture():
    _db_file = test_data / "test_db" / f"empty_db.db"
    _db = open_database(_db_file)
    assert isinstance(_db, SqliteDatabase)

    with _db.atomic() as transaction:
        yield _db
        transaction.rollback()
    _db.close()


def get_basic_dike_traject_info() -> DikeTrajectInfo:
    """
    Gets a basic dike traject info entity.

    Returns:
        DikeTrajectInfo: The created dike traject info entity in the database.
    """
    return DikeTrajectInfo.create(traject_name="123")


def get_basic_section_data() -> SectionData:
    """
    Gets a basic section data entity.

    Returns:
        SectionData: The created section data entity in the database.
    """
    _test_dike_traject = get_basic_dike_traject_info()
    return SectionData.create(
        dike_traject=_test_dike_traject,
        section_name="TestSection",
        meas_start=2.4,
        meas_end=4.2,
        section_length=123,
        in_analysis=True,
        crest_height=24,
        annual_crest_decline=42,
    )


def get_basic_mechanism_per_section() -> MechanismPerSection:
    """
    Gets a basic mechanism per section entity.

    Returns:
        MechanismPerSection: The created mechanism per section entity in the database.
    """
    _test_section = get_basic_section_data()

    _mechanism = Mechanism.create(name="OVERFLOW")
    return MechanismPerSection.create(section=_test_section, mechanism=_mechanism)


def get_basic_computation_scenario() -> ComputationScenario:
    """
    Gets a basic computation scenario entity.

    Returns:
        ComputationScenario: The created computation scenario entity in the database.
    """
    _mech_per_section = get_basic_mechanism_per_section()

    _computation_type = ComputationType.create(name="TestComputation")
    return ComputationScenario.create(
        mechanism_per_section=_mech_per_section,
        computation_type=_computation_type,
        computation_name="Test Computation",
        scenario_name="test_name",
        scenario_probability=0.42,
        probability_of_failure=0.24,
    )


def get_basic_measure_type() -> MeasureType:
    """
    Gets a basic measure type entity.

    Returns:
        MeasureType: The created measure type entity in the database.
    """
    return MeasureType.create(name="TestMeasureType")


def get_basic_combinable_type() -> CombinableType:
    """
    Gets a basic combinable type entity.

    Returns:
        CombinableType: The created combinable type entity in the database.
    """
    return CombinableType.create(name="TestCombinableType")


def get_basic_measure() -> Measure:
    """
    Gets a basic measure entity.

    Returns:
        Measure: The created measure entity in the database.
    """
    _test_measure_type = get_basic_measure_type()
    _test_combinable_type = get_basic_combinable_type()
    return Measure.create(
        measure_type=_test_measure_type,
        combinable_type=_test_combinable_type,
        name="TestMeasure",
        year=20,
    )


def get_basic_measure_per_section() -> MeasurePerSection:
    """
    Gets a basic measure per section entity.

    Returns:
        MeasurePerSection: The created measure per section entity in the database.
    """
    _test_section = get_basic_section_data()
    _test_measure = get_basic_measure()
    return MeasurePerSection.create(
        section=_test_section,
        measure=_test_measure,
    )


def get_basic_measure_result() -> MeasureResult:
    """
    Gets a basic measure result entity.

    Returns:
        MeasureResult: The created measure result entity in the database.
    """
    _test_measure_per_section = get_basic_measure_per_section()
    return MeasureResult.create(
        beta=3.1234,
        time=0.0,
        cost=100,
        measure_per_section=_test_measure_per_section,
    )
