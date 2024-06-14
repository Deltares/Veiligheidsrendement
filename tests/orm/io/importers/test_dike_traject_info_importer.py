import math

import pytest
from peewee import SqliteDatabase

from vrtool.common.dike_traject_info import DikeTrajectInfo as VrtoolDikeTrajectInfo
from vrtool.orm.io.importers.dike_traject_info_importer import DikeTrajectInfoImporter
from vrtool.orm.io.importers.orm_importer_protocol import OrmImporterProtocol
from vrtool.orm.models.dike_traject_info import DikeTrajectInfo as DikeTrajectInfo


class TestDikeTrajectInfoImporter:
    def test_initialize(self):
        _importer = DikeTrajectInfoImporter()
        assert isinstance(_importer, DikeTrajectInfoImporter)
        assert isinstance(_importer, OrmImporterProtocol)

    def test_import_orm(self, empty_db_context: SqliteDatabase):
        # 1. Define test data.
        with empty_db_context.atomic() as transaction:
            _orm_dike_traject_info = DikeTrajectInfo.create(
                traject_name="16-1",
                omega_piping=0.25,
                omega_stability_inner=0.04,
                omega_overflow=0.24,
                a_piping=None,
                b_piping=300,
                a_stability_inner=0.033,
                b_stability_inner=50,
                beta_max=3.7190164854556804,
                p_max=0.0001,
                flood_damage=None,
                traject_length=None,
            )
            transaction.commit()

        _importer = DikeTrajectInfoImporter()

        # 2. Run test.
        _dike_traject_info = _importer.import_orm(_orm_dike_traject_info)

        # 3. Verify final expectations.
        assert isinstance(_dike_traject_info, VrtoolDikeTrajectInfo)
        assert _dike_traject_info.traject_name == "16-1"
        assert _dike_traject_info.omegaPiping == 0.25
        assert _dike_traject_info.omegaStabilityInner == 0.04
        assert _dike_traject_info.omegaOverflow == 0.24
        assert math.isnan(_dike_traject_info.aPiping)
        assert _dike_traject_info.bPiping == 300
        assert _dike_traject_info.aStabilityInner == 0.033
        assert _dike_traject_info.bStabilityInner == 50
        assert _dike_traject_info.beta_max == pytest.approx(3.7190164854556804)
        assert _dike_traject_info.Pmax == 0.0001
        assert math.isnan(_dike_traject_info.FloodDamage)
        assert math.isnan(_dike_traject_info.TrajectLength)

    def test_import_orm_without_model_raises_value_error(self):
        # 1. Define test data.
        _importer = DikeTrajectInfoImporter()
        _expected_mssg = "No valid value given for DikeTrajectInfo."

        # 2. Run test.
        with pytest.raises(ValueError) as value_error:
            _importer.import_orm(None)

        # 3. Verify expectations.
        assert str(value_error.value) == _expected_mssg
