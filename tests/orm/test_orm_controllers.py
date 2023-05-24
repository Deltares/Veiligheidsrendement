import shutil

import pytest
from peewee import SqliteDatabase

from tests import test_data, test_results
from vrtool.defaults.vrtool_config import VrtoolConfig
from vrtool.flood_defence_system.dike_traject import DikeTraject
from vrtool.orm.models import *
from vrtool.orm.orm_controllers import (
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
    characteristic_point_type = [
        "BIT", "BUT", "BUK", "BIK", "EBL", "BBL"
    ]

    profile_points = [
        dict(x_coordinate=47.0,y_coordinate= 5.104),
        dict(x_coordinate=-17.0,y_coordinate= 4.996),
        dict(x_coordinate=0, y_coordinate=10.939),
        dict(x_coordinate=3.5, y_coordinate=10.937),
        dict(x_coordinate=42.0,y_coordinate= 5.694),
        dict(x_coordinate=25.0,y_coordinate= 6.491),
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
        if _db_file.parent.exists():
            shutil.rmtree(_db_file.parent)

        initialize_database(_db_file)

        # 2. Define models.
        _dike_traject_info: DikeTrajectInfo = DikeTrajectInfo.create(
            **DummyModelsData.dike_traject_info
        )
        _dike_traject_info.save()

        _dike_section: SectionData = SectionData.create(
            **(dict(dike_traject=_dike_traject_info) | DummyModelsData.section_data)
        )
        _dike_section.save()

        for _m_dict in DummyModelsData.mechanism_data:
            _mechanism = Mechanism.create(**_m_dict)
            _mechanism.save()
            _mechanism_section = MechanismPerSection.create(
                mechanism=_mechanism, section=_dike_section
            )
            _mechanism_section.save()

        for _b_dict in DummyModelsData.buildings_data:
            Buildings.create(**(_b_dict | dict(section_data=_dike_section))).save()
        
        for idx, _p_point in enumerate(DummyModelsData.profile_points):
            _c_point = CharacteristicPointType.create(**dict(name=DummyModelsData.characteristic_point_type[idx]))
            _c_point.save()
            ProfilePoint.create(**(_p_point | dict(section_data=_dike_section, profile_point_type=_c_point))).save()

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
        assert any(SectionData.select())
        _section_data: SectionData = SectionData.get_by_id(1)
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

    def test_import_dike_traject(self):
        # 1. Define test data.
        _db_file = test_data / "test_db" / "vrtool_db.db"
        assert _db_file.is_file()

        _config = VrtoolConfig(input_database_path=_db_file, traject="16-1")

        # 2. Run test.
        _dike_traject = get_dike_traject(_config)

        # 3. Verify expectations.
        assert isinstance(_dike_traject, DikeTraject)
        assert len(_dike_traject.sections) == 1
