from vrtool.common.enums.measure_type_enum import MeasureTypeEnum
from vrtool.decision_making.measures.measure_protocol import MeasureProtocol
from vrtool.decision_making.solutions import Solutions
from vrtool.defaults.vrtool_config import VrtoolConfig
from vrtool.flood_defence_system.dike_section import DikeSection
from vrtool.orm.io.importers.decision_making.measure_importer import MeasureImporter
from vrtool.orm.io.importers.orm_importer_protocol import OrmImporterProtocol
from vrtool.orm.models.measure import Measure as OrmMeasure
from vrtool.orm.models.measure_per_section import MeasurePerSection
from vrtool.orm.models.measure_type import MeasureType
from vrtool.orm.models.orm_base_model import OrmBaseModel
from vrtool.orm.models.section_data import SectionData


class SolutionsImporter(OrmImporterProtocol):
    def __init__(self, vrtool_config: VrtoolConfig, dike_section: DikeSection) -> None:
        if not vrtool_config:
            raise ValueError("{} not provided.".format(VrtoolConfig.__name__))
        if not dike_section:
            raise ValueError("{} not provided.".format(DikeSection.__name__))

        self._config = vrtool_config
        self._dike_section = dike_section

    def _import_measures(self, orm_measures: list[OrmMeasure]) -> list[MeasureProtocol]:
        _measure_importer = MeasureImporter(self._config)
        return list(map(_measure_importer.import_orm, orm_measures))

    @staticmethod
    def set_solution_measure_table(solution: Solutions):
        """
        Sets the measure table for a given imported solution.

        Args:
            solution (Solutions): Entity whose `measure_table` property needs to be set.
        """
        _combinables = []
        _partials = []
        for i, measure in enumerate(solution.measures):
            solution.measure_table.loc[i] = [
                str(measure.parameters["ID"]),
                measure.parameters["Name"],
            ]
            # also add the potential combined solutions up front
            if measure.parameters["Class"] == "combinable":
                _combinables.append(
                    (measure.parameters["ID"], measure.parameters["Name"])
                )
            if measure.parameters["Class"] == "partial":
                _partials.append((measure.parameters["ID"], measure.parameters["Name"]))
        count = 0
        for i in range(0, len(_partials)):
            for j in range(0, len(_combinables)):
                solution.measure_table.loc[count + len(solution.measures) + 1] = [
                    str(_partials[i][0]) + "+" + str(_combinables[j][0]),
                    str(_partials[i][1]) + "+" + str(_combinables[j][1]),
                ]
                count += 1

    def import_orm(self, orm_model: OrmBaseModel) -> Solutions:
        if not orm_model:
            raise ValueError(f"No valid value given for {SectionData.__name__}.")

        if self._dike_section.name != orm_model.section_name:
            raise ValueError(
                "The provided SectionData ({}) does not match the given DikeSection ({}).".format(
                    orm_model.section_name, self._dike_section.name
                )
            )

        _solutions = Solutions(self._dike_section, self._config)
        _solutions.measures = self._import_measures(
            list(
                OrmMeasure.select()
                .join_from(OrmMeasure, MeasurePerSection)
                .join_from(OrmMeasure, MeasureType)
                .where(
                    (MeasurePerSection.section == orm_model)
                    & (MeasureType.name != MeasureTypeEnum.CUSTOM.legacy_name)
                )
            )
        )
        self.set_solution_measure_table(_solutions)
        return _solutions
