from vrtool.decision_making.measures.measure_base import MeasureBase

from vrtool.decision_making.solutions import Solutions
from vrtool.flood_defence_system.dike_section import DikeSection
from vrtool.orm.io.importers.measure_importer import MeasureImporter

from vrtool.orm.io.importers.orm_importer_protocol import OrmImporterProtocol
from vrtool.defaults.vrtool_config import VrtoolConfig
from vrtool.orm.models.measure import Measure as OrmMeasure
from vrtool.orm.models.section_data import SectionData
from vrtool.orm.models.measure_per_section import MeasurePerSection


class SolutionImporter(OrmImporterProtocol):

    def __init__(self, vrtool_config: VrtoolConfig, dike_section: DikeSection) -> None:
        self._config = vrtool_config
        self._dike_section = dike_section

    def _import_measures(self, orm_measures: list[OrmMeasure]) -> list[MeasureBase]:
        _measure_importer = MeasureImporter(self._config, self._dike_section)
        return list(map(_measure_importer.import_orm, orm_measures))

    def import_orm(self, orm_model: SectionData) -> Solutions:
        _solutions = Solutions(self._dike_section, self._config)
        _solutions.measures = self._import_measures(orm_model.measures_per_section.select(MeasurePerSection.measure))
        return  _solutions