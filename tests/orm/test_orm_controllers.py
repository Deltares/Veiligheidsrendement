import shutil
from pathlib import Path

import pandas as pd
import pytest
from peewee import SqliteDatabase

import vrtool.orm.models as orm_models
from tests import test_data, test_results
from tests.orm import get_basic_mechanism_per_section
from vrtool.common.dike_traject_info import DikeTrajectInfo
from vrtool.common.hydraulic_loads.load_input import LoadInput
from vrtool.decision_making.solutions import Solutions
from vrtool.defaults.vrtool_config import VrtoolConfig
from vrtool.flood_defence_system.dike_section import DikeSection
from vrtool.flood_defence_system.dike_traject import DikeTraject
from vrtool.flood_defence_system.failure_mechanism_collection import (
    FailureMechanismCollection,
)
from vrtool.flood_defence_system.mechanism_reliability_collection import (
    MechanismReliabilityCollection,
)
from vrtool.flood_defence_system.section_reliability import SectionReliability
from vrtool.orm.orm_controllers import (
    export_results_safety_assessment,
    get_dike_section_solutions,
    get_dike_traject,
    initialize_database,
    open_database,
)
from vrtool.run_workflows.safety_workflow.results_safety_assessment import (
    ResultsSafetyAssessment,
)


class DummyModelsData:

    dike_traject_info = dict(
        traject_name="16-1",
        omega_piping=0.25,
        omega_stability_inner=0.04,
        omega_overflow=0.24,
        a_piping=float("nan"),
        b_piping=300,
        a_stability_inner=0.033,
        b_stability_inner=50,
        beta_max=0.01,
        p_max=0.0001,
        flood_damage=float("nan"),
        traject_length=0.0,
    )
    section_data = dict(
        section_name="section_one",
        dijkpaal_start="start_point",
        dijkpaal_end="end_point",
        meas_start=2.4,
        meas_end=4.2,
        section_length=123,
        in_analysis=True,
        crest_height=1.0,
        annual_crest_decline=2.0,
        cover_layer_thickness=3.0,
        pleistocene_level=4.0,
    )
    mechanism_data = [dict(name="a_mechanism"), dict(name="b_mechanism")]
    buildings_data = [
        dict(distance_from_toe=24, number_of_buildings=2),
        dict(distance_from_toe=42, number_of_buildings=1),
    ]
    characteristic_point_type = ["BIT", "BUT", "BUK", "BIK", "EBL", "BBL"]

    profile_points = [
        dict(x_coordinate=47.0, y_coordinate=5.104),
        dict(x_coordinate=-17.0, y_coordinate=4.996),
        dict(x_coordinate=0, y_coordinate=10.939),
        dict(x_coordinate=3.5, y_coordinate=10.937),
        dict(x_coordinate=42.0, y_coordinate=5.694),
        dict(x_coordinate=25.0, y_coordinate=6.491),
    ]


class TestOrmControllers:
    def test_create_db(self, request: pytest.FixtureRequest):
        # 1. Define test data.
        _db_file = test_results / request.node.name / "vrtool_db.db"
        if _db_file.parent.exists():
            shutil.rmtree(_db_file.parent)

        # 2. Run test.
        initialize_database(_db_file)

        # 3. Verify expectations.
        assert _db_file.exists()

    @pytest.mark.skipif(
        condition=(test_data.joinpath("test_db\\vrtool_db.db").exists()),
        reason="Test database already exists. Won't overwrite.",
    )
    def test_create_db_with_data(self):
        # 1. Define database file.
        _db_file = test_data / "test_db" / "vrtool_db.db"
        if _db_file.exists():
            _db_file.unlink()

        initialize_database(_db_file)

        # 2. Define models.
        _dike_traject_info: orm_models.DikeTrajectInfo = (
            orm_models.DikeTrajectInfo.create(**DummyModelsData.dike_traject_info)
        )
        _dike_traject_info.save()

        _dike_section: orm_models.SectionData = orm_models.SectionData.create(
            **(dict(dike_traject=_dike_traject_info) | DummyModelsData.section_data)
        )
        _dike_section.save()

        for _m_dict in DummyModelsData.mechanism_data:
            _mechanism = orm_models.Mechanism.create(**_m_dict)
            _mechanism.save()
            _mechanism_section = orm_models.MechanismPerSection.create(
                mechanism=_mechanism, section=_dike_section
            )
            _mechanism_section.save()

        for _b_dict in DummyModelsData.buildings_data:
            orm_models.Buildings.create(
                **(_b_dict | dict(section_data=_dike_section))
            ).save()

        for idx, _p_point in enumerate(DummyModelsData.profile_points):
            _c_point = orm_models.CharacteristicPointType.create(
                **dict(name=DummyModelsData.characteristic_point_type[idx])
            )
            _c_point.save()
            orm_models.ProfilePoint.create(
                **(
                    _p_point
                    | dict(section_data=_dike_section, profile_point_type=_c_point)
                )
            ).save()

        # 3. Save tables.
        assert _db_file.exists()

    def test_open_database(self):
        # 1. Define test data.
        _db_file = test_data / "test_db" / "vrtool_db.db"
        assert _db_file.is_file()
        _expected_data = DummyModelsData.section_data

        # 2. Run test.
        _sql_db = open_database(_db_file)

        # 3. Verify expectations
        assert isinstance(_sql_db, SqliteDatabase)
        assert any(orm_models.SectionData.select())
        _section_data: orm_models.SectionData = orm_models.SectionData.get_by_id(1)
        assert _section_data.section_name == _expected_data["section_name"]
        assert _section_data.dijkpaal_start == _expected_data["dijkpaal_start"]
        assert _section_data.dijkpaal_end == _expected_data["dijkpaal_end"]
        assert _section_data.meas_start == _expected_data["meas_start"]
        assert _section_data.meas_end == _expected_data["meas_end"]
        assert _section_data.section_length == _expected_data["section_length"]
        assert _section_data.in_analysis == _expected_data["in_analysis"]
        assert _section_data.crest_height == _expected_data["crest_height"]
        assert (
            _section_data.annual_crest_decline == _expected_data["annual_crest_decline"]
        )
        assert (
            _section_data.cover_layer_thickness
            == _expected_data["cover_layer_thickness"]
        )
        assert _section_data.pleistocene_level == _expected_data["pleistocene_level"]

    def test_open_database_when_file_doesnot_exist_raises_value_error(
        self, request: pytest.FixtureRequest
    ):
        # 1. Define test data.
        _db_file = test_results / request.node.name / "vrtool_db.db"
        assert not _db_file.exists()

        # 2. Run test.
        with pytest.raises(ValueError) as exc_err:
            open_database(_db_file)

        # 3. Verify expectations
        assert str(exc_err.value) == "No file was found at {}".format(_db_file)

    @pytest.fixture
    def database_vrtool_config(self, request: pytest.FixtureRequest) -> VrtoolConfig:
        # 1. Define test data.
        _db_file = test_data / "test_db" / "with_valid_data.db"
        assert _db_file.is_file()

        _vrtool_config = VrtoolConfig(
            input_database_path=_db_file, traject="38-1", input_directory=test_data
        )
        _vrtool_config.output_directory = test_results.joinpath(request.node.name)
        if _vrtool_config.output_directory.exists():
            shutil.rmtree(_vrtool_config.output_directory)

        yield _vrtool_config

    def test_get_dike_traject(self, database_vrtool_config: VrtoolConfig):
        # 2. Run test.
        _dike_traject = get_dike_traject(database_vrtool_config)

        # 3. Verify expectations.
        assert isinstance(_dike_traject, DikeTraject)
        assert len(_dike_traject.sections) == 60

        def check_section_reliability(section: DikeSection):
            _recognized_keys = [
                "Overflow",
                "Piping",
                "StabilityInner",
            ]

            def check_key_value(key_value):
                assert (
                    key_value[0] in _recognized_keys
                ), "Mechanism {} not recognized.".format(key_value[0])
                assert isinstance(key_value[1], MechanismReliabilityCollection)

            assert isinstance(section.section_reliability, SectionReliability)
            assert isinstance(section.section_reliability.load, LoadInput)
            assert isinstance(
                section.section_reliability.failure_mechanisms,
                FailureMechanismCollection,
            )

            all(
                map(
                    check_key_value,
                    section.section_reliability.failure_mechanisms._failure_mechanisms.items(),
                )
            )

        assert all(any(_ds.mechanism_data.items()) for _ds in _dike_traject.sections)
        all(map(check_section_reliability, _dike_traject.sections))

    def test_get_dike_section_solutions(self, database_vrtool_config: VrtoolConfig):

        # 1. Define test data.
        database_vrtool_config.T = [0]
        _general_info = DikeTrajectInfo(traject_name="Dummy")
        _dike_section = DikeSection()
        _dike_section.name = "01A"
        _dike_section.Length = 359.0

        # Stability Inner
        _water_load_input = LoadInput([])
        _water_load_input.input["d_cover"] = None
        _water_load_input.input["beta"] = 42.24
        _stability_inner_collection = MechanismReliabilityCollection(
            "StabilityInner", "combinable", database_vrtool_config.T, 2023, 2025
        )
        _stability_inner_collection.Reliability["0"].Input = _water_load_input
        _dike_section.section_reliability.load = _water_load_input
        _dike_section.section_reliability.failure_mechanisms._failure_mechanisms[
            "StabilityInner"
        ] = _stability_inner_collection

        # Initial Geometry
        def _to_record(geom_item: tuple[str, list[int]]) -> dict:
            return dict(type=geom_item[0], x=geom_item[1][0], z=geom_item[1][1])

        _initial_geom = {
            "BUT": [-17.0, 4.996],
            "BUK": [0, 10.939],
            "BIK": [3.5, 10.937],
            "BBL": [25.0, 6.491],
            "EBL": [42.0, 5.694],
            "BIT": [47.0, 5.104],
        }
        _dike_section.InitialGeometry = pd.DataFrame.from_records(
            map(_to_record, _initial_geom.items())
        )

        # Mechanism data
        _dike_section.mechanism_data["StabilityInner"] = [
            ("RW000", "SIMPLE"),
            "combinable",
        ]
        # 2. Run test.
        _solutions = get_dike_section_solutions(
            database_vrtool_config, _dike_section, _general_info
        )

        # 3. Verify expectations.
        assert isinstance(_solutions, Solutions)
        assert any(_solutions.measures)

    @pytest.fixture
    def export_database(self, request: pytest.FixtureRequest) -> Path:
        _db_file = test_data / "test_db" / f"empty_db.db"
        _output_dir = test_results.joinpath(request.node.name)
        if _output_dir.exists():
            shutil.rmtree(_output_dir)
        _output_dir.mkdir(parents=True)
        _test_db_file = _output_dir.joinpath("test_db.db")
        shutil.copyfile(_db_file, _test_db_file)

        _connected_db = open_database(_test_db_file)
        _connected_db.close()
        yield _test_db_file
        # Make sure it's closed.
        # Perhaps during test something fails and does not get to close)
        _connected_db.close()

    def test_export_results_safety_assessment_given_valid_data(
        self, export_database: Path
    ):
        # 1. Define test data.
        _connected_db = open_database(export_database)
        _test_mechanism_per_section = get_basic_mechanism_per_section()
        _connected_db.close()
        _test_section_data = _test_mechanism_per_section.section

        # Dike Section and Dike Traject.
        _reliability_df = pd.DataFrame(
            [4.2, 2.4],
            columns=["42"],
            index=[_test_mechanism_per_section.mechanism.name, "Section"],
        )
        _dummy_section = DikeSection()
        _dummy_section.name = _test_section_data.section_name
        _dummy_section.TrajectInfo = DikeTrajectInfo(
            traject_name=_test_section_data.dike_traject.traject_name
        )
        _dummy_section.section_reliability.SectionReliability = _reliability_df
        _test_traject = DikeTraject()
        _test_traject.sections = [_dummy_section]

        # Safety assessment.
        _safety_assessment = ResultsSafetyAssessment()
        _safety_assessment.vr_config = VrtoolConfig(input_database_path=export_database)
        _safety_assessment.selected_traject = _test_traject

        assert not any(orm_models.AssessmentSectionResult.select())
        assert not any(orm_models.AssessmentMechanismResult.select())

        # 2. Run test.
        export_results_safety_assessment(_safety_assessment)

        # 3. Verify final expectations.
        assert any(
            orm_models.AssessmentSectionResult.select().where(
                (orm_models.AssessmentSectionResult.section_data == _test_section_data)
                & (orm_models.AssessmentSectionResult.beta == 2.4)
                & (orm_models.AssessmentSectionResult.time == 42)
            )
        )
        assert any(
            orm_models.AssessmentMechanismResult.select().where(
                (
                    orm_models.AssessmentMechanismResult.mechanism_per_section
                    == _test_mechanism_per_section
                )
                & (orm_models.AssessmentMechanismResult.beta == 4.2)
                & (orm_models.AssessmentMechanismResult.time == 42)
            )
        )
