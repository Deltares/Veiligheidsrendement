import shutil

import pandas as pd
import pytest
from peewee import SqliteDatabase

import vrtool.orm.models as orm_models
from tests import test_data, test_results
from tests.orm import get_basic_dike_traject_info
from vrtool.common.hydraulic_loads.load_input import LoadInput
from vrtool.decision_making.solutions import Solutions
from vrtool.defaults.vrtool_config import VrtoolConfig
from vrtool.flood_defence_system.dike_section import DikeSection
from vrtool.flood_defence_system.dike_traject import DikeTraject
from vrtool.flood_defence_system.failure_mechanism_collection import (
    FailureMechanismCollection,
)
from vrtool.flood_defence_system.mechanism_reliability import MechanismReliability
from vrtool.flood_defence_system.mechanism_reliability_collection import (
    MechanismReliabilityCollection,
)
from vrtool.flood_defence_system.section_reliability import SectionReliability
from vrtool.orm.models.assessment_mechanism_result import AssessmentMechanismResult
from vrtool.orm.models.assessment_section_result import AssessmentSectionResult
from vrtool.orm.models.dike_traject_info import DikeTrajectInfo
from vrtool.orm.models.mechanism import Mechanism
from vrtool.orm.models.mechanism_per_section import MechanismPerSection
from vrtool.orm.models.section_data import SectionData
from vrtool.orm.orm_controllers import (
    clear_assessment_results,
    get_dike_section_solutions,
    get_dike_traject,
    initialize_database,
    open_database,
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
        _general_info = DikeTrajectInfo()
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

    def test_clear_assessment_results_clears_all_results(
        self, request: pytest.FixtureRequest
    ):
        # Setup
        _db_file = test_results / request.node.name / "vrtool_db.db"
        if _db_file.parent.exists():
            shutil.rmtree(_db_file.parent)

        initialize_database(_db_file)
        test_db = open_database(_db_file)

        traject_info = get_basic_dike_traject_info()

        _mechanism_one = self._create_mechanism("mechanism 1")
        _mechanism_two = self._create_mechanism("mechanism 2")

        _section_one = self._create_basic_section_data(traject_info, "section 1")
        self._create_assessment_section_results(_section_one)
        mechanism_one_per_section_one = self._create_basic_mechanism_per_section(
            _section_one, _mechanism_one
        )
        self._create_assessment_mechanism_results(mechanism_one_per_section_one)
        mechanism_two_per_section_one = self._create_basic_mechanism_per_section(
            _section_one, _mechanism_two
        )
        self._create_assessment_mechanism_results(mechanism_two_per_section_one)

        _section_two = self._create_basic_section_data(traject_info, "section 2")
        self._create_assessment_section_results(_section_two)
        mechanism_one_per_section_two = self._create_basic_mechanism_per_section(
            _section_one, _mechanism_one
        )
        self._create_assessment_mechanism_results(mechanism_one_per_section_two)

        mechanism_two_per_section_two = self._create_basic_mechanism_per_section(
            _section_one, _mechanism_two
        )
        self._create_assessment_mechanism_results(mechanism_two_per_section_two)

        _vrtool_config = VrtoolConfig(input_database_path=_db_file)

        # Precondition
        assert any(AssessmentSectionResult.select())
        assert any(AssessmentMechanismResult.select())

        test_db.close()

        # Call
        clear_assessment_results(_vrtool_config)

        # Assert
        test_db = open_database(_db_file)

        assert not any(AssessmentSectionResult.select())
        assert not any(AssessmentMechanismResult.select())

        test_db.close()

    def _create_basic_section_data(
        self, traject_info: DikeTrajectInfo, section_name: str
    ) -> SectionData:
        return SectionData.create(
            dike_traject=traject_info,
            section_name=section_name,
            meas_start=2.4,
            meas_end=4.2,
            section_length=123,
            in_analysis=True,
            crest_height=24,
            annual_crest_decline=42,
        )

    def _create_assessment_section_results(self, section: SectionData) -> None:
        for i in range(2000, 2100, 10):
            AssessmentSectionResult.create(
                beta=i / 1000.0, time=i, section_data=section
            )

    def _create_mechanism(self, mechanism_name: str) -> Mechanism:
        return Mechanism.create(name=mechanism_name)

    def _create_basic_mechanism_per_section(
        self, section: SectionData, mechanism: Mechanism
    ) -> MechanismPerSection:
        return MechanismPerSection.create(section=section, mechanism=mechanism)

    def _create_assessment_mechanism_results(
        self, mechanism_per_section: MechanismPerSection
    ) -> None:
        for i in range(2000, 2100, 10):
            AssessmentMechanismResult.create(
                beta=i / 1000.0, time=i, mechanism_per_section=mechanism_per_section
            )
