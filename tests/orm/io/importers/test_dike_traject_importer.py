import pytest

from tests.orm import with_empty_db_fixture
from vrtool.common.dike_traject_info import DikeTrajectInfo
from vrtool.defaults.vrtool_config import VrtoolConfig
from vrtool.flood_defence_system.dike_traject import DikeTraject
from vrtool.orm.io.importers.dike_traject_importer import DikeTrajectImporter
from vrtool.orm.io.importers.orm_importer_protocol import OrmImporterProtocol
from vrtool.orm.models.dike_traject_info import DikeTrajectInfo
from vrtool.orm.models.dike_traject_info import DikeTrajectInfo as OrmDikeTrajectInfo


class TestDikeTrajectImporter:
    @with_empty_db_fixture
    def test_initialize(self):
        config = VrtoolConfig(input_directory=".")
        DikeTrajectInfo.create(traject_name="123")
        _importer = DikeTrajectImporter(config)
        assert isinstance(_importer, DikeTrajectImporter)
        assert isinstance(_importer, OrmImporterProtocol)

    def test_import_orm(self, empty_db_fixture):
        # 1. Define test data.
        config = VrtoolConfig(input_directory=".")
        DikeTrajectInfo.create(traject_name="123")
        _importer = DikeTrajectImporter(config)

        # 2. Run test.
        _dike_traject = _importer.import_orm(OrmDikeTrajectInfo.get_by_id(1))

        # 3. Verify final expectations.
        assert isinstance(_dike_traject, DikeTraject)

        assert _dike_traject.t_0 == config.t_0
        assert _dike_traject.T == config.T

        assert _dike_traject.general_info.traject_name == "123"

    def test_import_orm_without_model_raises_value(self, empty_db_fixture):
        # 1. Define test data.
        config = VrtoolConfig(input_directory=".")
        _importer = DikeTrajectImporter(config)
        _expected_mssg = "No valid value given for DikeTrajectInfo."

        # 2. Run test.
        with pytest.raises(ValueError) as value_error:
            _importer.import_orm(None)

        # 3. Verify expectations.
        assert str(value_error.value) == _expected_mssg
