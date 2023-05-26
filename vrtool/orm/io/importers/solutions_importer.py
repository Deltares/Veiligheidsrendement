from vrtool.decision_making.measures.measure_base import MeasureBase
from vrtool.decision_making.solutions import Solutions
from vrtool.defaults.vrtool_config import VrtoolConfig
from vrtool.flood_defence_system.dike_section import DikeSection
from vrtool.orm.io.importers.measure_importer import MeasureImporter
from vrtool.orm.io.importers.orm_importer_protocol import OrmImporterProtocol
from vrtool.orm.models.measure import Measure as OrmMeasure
from vrtool.orm.models.measure_per_section import MeasurePerSection
from vrtool.orm.models.section_data import SectionData


class SolutionsImporter(OrmImporterProtocol):
    def __init__(self, vrtool_config: VrtoolConfig, dike_section: DikeSection) -> None:
        if not vrtool_config:
            raise ValueError("{} not provided.".format(VrtoolConfig.__name__))
        if not dike_section:
            raise ValueError("{} not provided.".format(DikeSection.__name__))

        self._config = vrtool_config
        self._dike_section = dike_section

    def _import_measures(self, orm_measures: list[OrmMeasure]) -> list[MeasureBase]:
        _measure_importer = MeasureImporter(self._config)
        return list(map(_measure_importer.import_orm, orm_measures))

    def import_orm(self, orm_model: SectionData) -> Solutions:

        if not orm_model:
            raise ValueError(f"No valid value given for {SectionData.__name__}.")

        if self._dike_section.name != orm_model.section_name:
            raise ValueError("The provided SectionData ({}) does not match the given DikeSection ({}).".format(orm_model.section_name, self._dike_section.name))

        _solutions = Solutions(self._dike_section, self._config)
        _solutions.measures = self._import_measures(
            list(OrmMeasure.select().join(MeasurePerSection).where(orm_model.id == MeasurePerSection.section_id))
        )
        return _solutions
