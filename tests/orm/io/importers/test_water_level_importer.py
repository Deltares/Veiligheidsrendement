from __future__ import annotations

from typing import Callable, Iterator

import pytest
from peewee import SqliteDatabase
from pytest import approx

from tests.orm import with_empty_db_context
from vrtool.orm.io.importers.orm_importer_protocol import OrmImporterProtocol
from vrtool.orm.io.importers.water_level_importer import WaterLevelImporter
from vrtool.orm.models.section_data import SectionData
from vrtool.orm.models.water_level_data import WaterlevelData

wl1 = [
    {
        "section_data_id": 1,
        "water_level": 2.0,
        "year": 2030,
        "beta": 2.2,
    },
]
wl2 = [
    {
        "section_data_id": 1,
        "water_level": 4.0,
        "year": 2030,
        "beta": 4.4,
    },
]
wl3 = [
    {
        "section_data_id": 1,
        "water_level": 2.3,
        "year": 2050,
        "beta": 2.2,
    },
]
wl4 = [
    {
        "section_data_id": 1,
        "water_level": 4.3,
        "year": 2050,
        "beta": 4.4,
    },
]


class TestWaterLevelImporter:
    def test_initialize_water_level_importer(self):
        _importer = WaterLevelImporter(42)
        assert isinstance(_importer, WaterLevelImporter)
        assert isinstance(_importer, OrmImporterProtocol)
        assert _importer.gridpoint == 42

    @with_empty_db_context
    def test_import_orm_without_no_water_level_data_doesnot_raise(
        self, get_orm_basic_dike_section: Callable[[], SectionData]
    ):
        # 1. Define test data.
        _importer = WaterLevelImporter(42)
        _section_data = get_orm_basic_dike_section()
        assert not any(_section_data.water_level_data_list)

        # 2. Run test.
        _load_input = _importer.import_orm(_section_data)

        # 3. Verify expectations.
        assert _load_input is None

    @pytest.fixture(name="water_level_importer_orm_dike_section")
    def _get_water_level_importer_orm_dike_section_fixture(
        self,
        request: pytest.FixtureRequest,
        empty_db_context: SqliteDatabase,
        get_orm_basic_dike_section: Callable[[], SectionData],
    ) -> Iterator[SectionData]:
        with empty_db_context.atomic() as transaction:
            _section_data = get_orm_basic_dike_section()

            WaterlevelData.insert_many(request.param).execute()
            transaction.commit()
        yield _section_data

    @pytest.mark.parametrize(
        "water_level_importer_orm_dike_section",
        [pytest.param(wl1 + wl2, id="base case")],
        indirect=True,
    )
    def test_import_water_level(
        self, water_level_importer_orm_dike_section: SectionData
    ):
        # 1. Define test data.
        _importer = WaterLevelImporter(1000)

        # 2. Run test
        _load = _importer.import_orm(water_level_importer_orm_dike_section)

        # 3. Verify expectations.
        sd = _load.distribution[2030].getStandardDeviation()[0]
        assert sd == approx(2.0278)

    @pytest.mark.parametrize(
        "water_level_importer_orm_dike_section",
        [
            pytest.param(wl2 + wl4 + wl3 + wl1, id="with shuffle"),
            pytest.param(wl1 + wl2 + wl3 + wl4, id="without shuffle"),
        ],
        indirect=True,
    )
    def test_import_water_level_two_years(
        self,
        water_level_importer_orm_dike_section: SectionData,
    ):
        # 1. Define test data.
        _importer = WaterLevelImporter(1000)

        # 2. Run test
        _load = _importer.import_orm(water_level_importer_orm_dike_section)

        # 3. Verify expectations.
        sd2030 = _load.distribution[2030].getStandardDeviation()[0]
        assert sd2030 == approx(2.0278)
        sd2050 = _load.distribution[2050].getStandardDeviation()[0]
        assert sd2050 == approx(2.0278)
        mean2030 = _load.distribution[2030].getMean()[0]
        mean2050 = _load.distribution[2050].getMean()[0]
        diff = mean2050 - mean2030
        assert diff == approx(0.3)
