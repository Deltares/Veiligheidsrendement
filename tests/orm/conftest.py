import shutil
from pathlib import Path
from typing import Callable, Iterator, Optional

import pytest
from peewee import SqliteDatabase

from tests import get_clean_test_results_dir, test_data, test_results
from vrtool.common.dike_traject_info import DikeTrajectInfo
from vrtool.common.enums.mechanism_enum import MechanismEnum
from vrtool.defaults.vrtool_config import VrtoolConfig
from vrtool.flood_defence_system.dike_section import DikeSection
from vrtool.orm.models.combinable_type import CombinableType
from vrtool.orm.models.computation_scenario import ComputationScenario
from vrtool.orm.models.computation_type import ComputationType
from vrtool.orm.models.dike_traject_info import DikeTrajectInfo as OrmDikeTrajectInfo
from vrtool.orm.models.measure import Measure
from vrtool.orm.models.measure_per_section import MeasurePerSection
from vrtool.orm.models.measure_result import MeasureResult
from vrtool.orm.models.measure_type import MeasureType
from vrtool.orm.models.mechanism import Mechanism
from vrtool.orm.models.mechanism_per_section import MechanismPerSection
from vrtool.orm.models.section_data import SectionData
from vrtool.orm.orm_controllers import open_database


@pytest.fixture(name="persisted_database")
def get_persisted_database_fixture(
    request: pytest.FixtureRequest,
) -> Iterator[SqliteDatabase]:
    """
    Gets an empty database context with a valid scheme
    in a directory where it will be persisted after the test
    finalizes, allowing to inspect its results.
    This fixture's database is used when the database needs to be opened
    and closed during multiple times in a test.
    """
    # Create a results directory where to persist the database.
    _output_dir = test_results.joinpath(request.node.name)
    if _output_dir.exists():
        shutil.rmtree(_output_dir)
    _output_dir.mkdir(parents=True)
    _test_db_file = _output_dir.joinpath("test_db.db")

    # Copy the original `empty_db.db` into the output directory.
    _db_file = test_data.joinpath("test_db", "empty_db.db")
    shutil.copyfile(_db_file, _test_db_file)

    # Initialized its context.
    _connected_db = open_database(_test_db_file)
    _connected_db.close()

    yield _connected_db

    # Make sure it's closed.
    # Perhaps during test something fails and does not get to close
    if isinstance(_connected_db, SqliteDatabase) and not _connected_db.is_closed():
        _connected_db.close()


@pytest.fixture(name="empty_db_context", autouse=False)
def get_empty_db_context_fixture() -> Iterator[SqliteDatabase]:
    """
    Gets an empty database context with a valid scheme.
    This fixture DOES NOT allow to open and close during the test,
    as its transaction is already initialized.
    """
    _db_file = test_data.joinpath("test_db", "empty_db.db")
    _db = open_database(_db_file)
    assert isinstance(_db, SqliteDatabase)

    with _db.atomic() as transaction:
        yield _db
        transaction.rollback()
    _db.close()


@pytest.fixture(name="custom_measures_vrtool_config")
def get_vrtool_config_for_custom_measures_db(
    request: pytest.FixtureRequest,
) -> Iterator[VrtoolConfig]:
    """
    Retrieves a valid `VrtoolConfig` instance ready to run a database
    for / with custom measures.
    In order to use it the test needs to provide the database path by using
    a marker such as :
        @pytest.mark.fixture_database(Path//to//database)
    For now this test assumes the selected traject is '38-1'.
    """
    # 1. Define test data.
    _marker = request.node.get_closest_marker("fixture_database")
    if _marker is None:
        _test_db = request.param
    else:
        _test_db = _marker.args[0]

    _output_directory = get_clean_test_results_dir(request)

    # Create a copy of the database to avoid locking it
    # or corrupting its data.
    _copy_db = _output_directory.joinpath("vrtool_input.db")
    shutil.copyfile(_test_db, _copy_db)

    # Generate a custom `VrtoolConfig`
    _vrtool_config = VrtoolConfig(
        input_directory=_copy_db.parent,
        input_database_name=_copy_db.name,
        traject="38-1",
        output_directory=_output_directory,
        discount_rate=0.03,
    )
    assert _vrtool_config.input_database_path.is_file()

    yield _vrtool_config


# Factory methods


def _get_domain_basic_dike_traject_info() -> DikeTrajectInfo:
    """
    Returns a basic dike traject info generator from the Vrtool data model.
    """
    return DikeTrajectInfo(traject_name=123)


def _get_domain_basic_dike_section() -> DikeSection:
    _dike_section = DikeSection()
    _dike_section.name = "TestSection"
    _dike_section.TrajectInfo = _get_domain_basic_dike_traject_info()
    return _dike_section


def _get_orm_basic_dike_traject_info() -> DikeTrajectInfo:
    _domain_dike_traject_info = _get_domain_basic_dike_traject_info()
    return OrmDikeTrajectInfo.create(
        traject_name=_domain_dike_traject_info.traject_name
    )


def _get_orm_basic_dike_section() -> SectionData:
    _test_dike_traject = _get_orm_basic_dike_traject_info()
    return SectionData.create(
        dike_traject=_test_dike_traject,
        section_name=_get_domain_basic_dike_section().name,
        meas_start=2.4,
        meas_end=4.2,
        section_length=123,
        in_analysis=True,
        crest_height=24,
        annual_crest_decline=42,
    )


def _get_basic_measure_type(name: str = "TestMeasureType") -> MeasureType:
    return MeasureType.create(name=name)


def _get_basic_combinable_type() -> CombinableType:
    return CombinableType.create(name="TestCombinableType")


def _get_basic_measure() -> Measure:
    _test_measure_type = _get_basic_measure_type()
    _test_combinable_type = _get_basic_combinable_type()
    return Measure.create(
        measure_type=_test_measure_type,
        combinable_type=_test_combinable_type,
        name="TestMeasure",
        year=20,
    )


def _get_basic_measure_per_section() -> MeasurePerSection:
    _test_section = _get_orm_basic_dike_section()
    _test_measure = _get_basic_measure()
    return MeasurePerSection.create(
        section=_test_section,
        measure=_test_measure,
    )


# Factory fixtures (using factory methods)


@pytest.fixture(name="get_orm_basic_dike_traject_info")
def get_orm_basic_dike_traject_info_factory() -> Iterator[
    Callable[[], OrmDikeTrajectInfo]
]:
    """
    Gets a basic dike traject info entity generator method.
    """
    yield _get_orm_basic_dike_traject_info


@pytest.fixture(name="get_domain_basic_dike_section")
def get_domain_basic_dike_section_factory() -> Iterator[Callable[[], DikeSection]]:
    """
    Yields a basic dike section generator from the Vrtool data model.
    """
    yield _get_domain_basic_dike_section


@pytest.fixture(name="get_orm_basic_dike_section")
def get_orm_basic_section_data_factory() -> Iterator[Callable[[], SectionData]]:
    """
    Gets a basic section data entity generator.
    """
    yield _get_orm_basic_dike_section


@pytest.fixture(name="get_basic_mechanism_per_section")
def get_basic_mechanism_per_section_factory(
    get_orm_basic_dike_section: Callable[[], SectionData],
) -> Iterator[Callable[[], MechanismPerSection]]:
    """
    Gets a basic mechanism per section entity generator.
    """

    def get_basic_mechanism_per_section() -> MechanismPerSection:
        _test_section = get_orm_basic_dike_section()

        _mech_inst = Mechanism.create(name=MechanismEnum.OVERFLOW.name)
        return MechanismPerSection.create(section=_test_section, mechanism=_mech_inst)

    yield get_basic_mechanism_per_section


@pytest.fixture(name="get_basic_computation_scenario")
def get_basic_computation_scenario_factory(
    get_basic_mechanism_per_section: Callable[[], MechanismPerSection]
) -> Iterator[Callable[[], ComputationScenario]]:
    """
    Gets a basic computation scenario entity generator.
    """

    def get_basic_computation_scenario() -> ComputationScenario:
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

    yield get_basic_computation_scenario


@pytest.fixture(name="get_basic_measure_type")
def get_basic_measure_type_factory() -> Iterator[
    Callable[[Optional[str]], MeasureType]
]:
    """
    Gets a basic measure type entity generator.
    """

    yield _get_basic_measure_type


@pytest.fixture(name="get_basic_combinable_type")
def get_basic_combinable_type_factory() -> Iterator[Callable[[], CombinableType]]:
    """
    Gets a basic combinable type entity generator.
    """
    yield _get_basic_combinable_type


@pytest.fixture(name="get_basic_measure")
def get_basic_measure_factory() -> Iterator[Callable[[], Measure]]:
    """
    Gets a basic measure entity generator.
    """
    yield _get_basic_measure


@pytest.fixture(name="get_basic_measure_per_section")
def get_basic_measure_per_section_factory() -> Iterator[
    Callable[[], MeasurePerSection]
]:
    """
    Gets a basic measure per section entity generator.
    """
    yield _get_basic_measure_per_section


@pytest.fixture(name="get_basic_measure_result")
def get_basic_measure_result_factory(
    get_basic_measure_per_section: Callable[[], MeasurePerSection]
) -> Iterator[Callable[[], MeasureResult]]:
    """
    Gets a basic measure result entity generator.
    """

    def get_basic_measure_result() -> MeasureResult:
        _test_measure_per_section = get_basic_measure_per_section()
        return MeasureResult.create(
            beta=3.1234,
            time=0.0,
            cost=100,
            measure_per_section=_test_measure_per_section,
        )

    yield get_basic_measure_result
