import pandas as pd

from vrtool.flood_defence_system.dike_section import DikeSection
from vrtool.orm.io.importers.orm_importer_protocol import OrmImporterProtocol
from vrtool.orm.models.buildings import Buildings
from vrtool.orm.models.section_data import SectionData


class DikeSectionImporter(OrmImporterProtocol):

    def _import_buildings_list(self, buildings_list: list[Buildings]) -> pd.DataFrame:
        _buildings_data = [[_building.distance_from_toe, _building.number_of_buildings] for _building in buildings_list]
        return pd.DataFrame(_buildings_data, columns=['distancefromtoe', 'cumulative'])

    def import_orm(self, orm_model: SectionData) -> DikeSection:
        _dike_section = DikeSection()
        _dike_section.name = orm_model.section_name
        _dike_section.houses = self._import_buildings_list(orm_model.buildings_list)
        _dike_section.mechanism_data = {}
        for _mechanism_per_section in orm_model.mechanisms_per_section:
            _dike_section.mechanism_data[_mechanism_per_section.mechanism.name] = ()
        return _dike_section