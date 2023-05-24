import math
import pandas as pd
import pytest
from peewee import SqliteDatabase
from typing import Union

from tests.orm.integration import valid_data_db_fixture
from vrtool.common.dike_traject_info import DikeTrajectInfo
from vrtool.orm.models import DikeTrajectInfo as OrmDikeTrajectInfo
from vrtool.orm.io.importers.dike_traject_importer import DikeTrajectImporter
from vrtool.orm.models.section_data import SectionData


class TestDatabaseIntegration:
    def test_import_dike_traject_imports_all_data(
        self, valid_data_db_fixture: SqliteDatabase
    ):
        # Setup
        _importer = DikeTrajectImporter()
        _orm_dike_traject_info = OrmDikeTrajectInfo.get_by_id(1)

        # Call
        _dike_traject = _importer.import_orm(_orm_dike_traject_info)

        # Assert
        self._assert_dike_traject_info(
            _dike_traject.general_info, _orm_dike_traject_info
        )

        _orm_dike_sections = _orm_dike_traject_info.dike_sections.select().where(
            SectionData.in_analysis
        )
        assert len(_dike_traject.sections) == len(_orm_dike_sections)

    def _assert_dike_traject_info(
        self, actual: DikeTrajectInfo, expected: OrmDikeTrajectInfo
    ) -> None:
        assert actual.traject_name == expected.traject_name

        self._assert_float(actual.omegaPiping, expected.omega_piping)
        self._assert_float(actual.omegaOverflow, expected.omega_overflow)
        self._assert_float(actual.omegaStabilityInner, expected.omega_stability_inner)

        self._assert_float(actual.aPiping, expected.a_piping)
        self._assert_float(actual.bPiping, expected.b_piping)

        self._assert_float(actual.aStabilityInner, expected.a_stability_inner)
        self._assert_float(actual.bStabilityInner, expected.b_stability_inner)

        # Note that currently the value for beta max is a derived value and is therefore not asserted.
        self._assert_float(actual.Pmax, expected.p_max)

        self._assert_float(actual.FloodDamage, expected.flood_damage)
        self._assert_float(actual.TrajectLength, expected.traject_length)

    def _assert_float(self, actual: float, expected: Union[float, None]) -> None:
        if not expected:
            assert math.isnan(actual)
        else:
            assert actual == pytest.approx(expected)
