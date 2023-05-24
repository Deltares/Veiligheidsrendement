from peewee import JOIN

from vrtool.flood_defence_system.dike_section import DikeSection
from vrtool.flood_defence_system.dike_traject import DikeTraject
from vrtool.orm.io.importers.dike_section_importer import DikeSectionImporter
from vrtool.orm.io.importers.dike_traject_info_importer import DikeTrajectInfoImporter
from vrtool.orm.io.importers.orm_importer_protocol import OrmImporterProtocol
from vrtool.orm.models.dike_traject_info import DikeTrajectInfo as OrmDikeTrajectInfo
from vrtool.orm.models.mechanism import Mechanism
from vrtool.orm.models.mechanism_per_section import MechanismPerSection
from vrtool.orm.models.section_data import SectionData
from vrtool.defaults.vrtool_config import VrtoolConfig


class DikeTrajectImporter(OrmImporterProtocol):
    _vrtool_config: VrtoolConfig

    def __init__(self, vrtool_config: VrtoolConfig) -> None:
        self._vrtool_config = vrtool_config

    def _import_dike_section_list(
        self, orm_dike_section_list: list[SectionData]
    ) -> list[DikeSection]:
        _ds_importer = DikeSectionImporter(self._vrtool_config)
        return list(map(_ds_importer.import_orm, orm_dike_section_list))

    def _select_available_mechanisms(
        self, dike_traject_info: OrmDikeTrajectInfo
    ) -> list[Mechanism]:
        return list(
            Mechanism.select()
            .join(MechanismPerSection)
            .join(SectionData)
            .join(OrmDikeTrajectInfo)
            .where(
                (OrmDikeTrajectInfo.id == dike_traject_info.id)
                & (SectionData.in_analysis == True)
            )
        )

    def import_orm(self, orm_model: OrmDikeTrajectInfo) -> DikeTraject:
        if not orm_model:
            raise ValueError(f"No valid value given for {OrmDikeTrajectInfo.__name__}.")

        _dike_traject = DikeTraject()
        _dike_traject.general_info = DikeTrajectInfoImporter().import_orm(orm_model)

        # Currently it is assumed that all SectionData present in a db belongs to whatever traject name has been provided.
        _selected_sections = orm_model.dike_sections.select().where(
            SectionData.in_analysis == True
        )
        _dike_traject.sections = self._import_dike_section_list(_selected_sections)
        _mechanisms = self._select_available_mechanisms(orm_model)
        _dike_traject.mechanism_names = list(set([_m.name for _m in _mechanisms]))
        _dike_traject.assessment_plot_years = self._vrtool_config.assessment_plot_years
        _dike_traject.flip_traject = self._vrtool_config.flip_traject
        _dike_traject.t_0 = self._vrtool_config.t_0
        _dike_traject.T = self._vrtool_config.T

        return _dike_traject
