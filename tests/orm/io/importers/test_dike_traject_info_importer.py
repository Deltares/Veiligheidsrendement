from vrtool.orm.io.importers.orm_importer_protocol import OrmImporterProtocol
from vrtool.orm.io.importers.dike_traject_info_importer import DikeTrajectInfoImporter

class TestDikeTrajectInfoImporter:

    def test_initialize(self):
        _importer = DikeTrajectInfoImporter()
        assert isinstance(_importer, DikeTrajectInfoImporter)
        assert isinstance(_importer, OrmImporterProtocol)