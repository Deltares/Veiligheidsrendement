from vrtool.orm.io.importers.dike_traject_importer import DikeTrajectImporter
from vrtool.orm.io.importers.orm_importer_protocol import OrmImporterProtocol


class TestDikeTrajectInfoImporter:
    def test_initialize(self):
        _importer = DikeTrajectImporter()
        assert isinstance(_importer, DikeTrajectImporter)
        assert isinstance(_importer, OrmImporterProtocol)
