from vrtool.common.enums.mechanism_enum import MechanismEnum
from vrtool.defaults.vrtool_config import VrtoolConfig
from vrtool.flood_defence_system.dike_section import DikeSection
from vrtool.flood_defence_system.dike_traject import DikeTraject
from vrtool.orm.io.importers.dike_section_importer import DikeSectionImporter
from vrtool.orm.io.importers.dike_traject_info_importer import DikeTrajectInfoImporter
from vrtool.orm.io.importers.orm_importer_protocol import OrmImporterProtocol
from vrtool.orm.models.dike_traject_info import DikeTrajectInfo as OrmDikeTrajectInfo
from vrtool.orm.models.mechanism import Mechanism
from vrtool.orm.models.mechanism_per_section import MechanismPerSection
from vrtool.orm.models.section_data import SectionData


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
        def to_enum(mechanism_inst: Mechanism) -> MechanismEnum:
            return MechanismEnum.get_enum(mechanism_inst.name)

        # Just define the query but don't instantiate it yet.
        _mechanism_selection_query = (
            Mechanism.select()
            .join(MechanismPerSection)
            .join(SectionData)
            .join(OrmDikeTrajectInfo)
            .where(
                (OrmDikeTrajectInfo.id == dike_traject_info.id)
                & (SectionData.in_analysis == True)
            )
        )
        # Map selection to `MechanismEnum`
        # -> (set) remove duplicates
        # -> return as list
        return list(set(map(to_enum, _mechanism_selection_query)))

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
        for _section in _dike_traject.sections:
            _section.TrajectInfo = _dike_traject.general_info
        _dike_traject.mechanisms = self._select_available_mechanisms(orm_model)
        _dike_traject.t_0 = self._vrtool_config.t_0
        _dike_traject.T = self._vrtool_config.T

        return _dike_traject
