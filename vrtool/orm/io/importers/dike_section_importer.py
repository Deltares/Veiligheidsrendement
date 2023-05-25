from __future__ import annotations

from pathlib import Path

import pandas as pd

from vrtool.defaults.vrtool_config import VrtoolConfig
from vrtool.flood_defence_system.dike_section import DikeSection
from vrtool.orm.io.importers.geometry_importer import GeometryImporter
from vrtool.orm.io.importers.orm_importer_protocol import OrmImporterProtocol
from vrtool.orm.models.buildings import Buildings
from vrtool.orm.models.profile_point import ProfilePoint
from vrtool.orm.models.section_data import SectionData


class DikeSectionImporter(OrmImporterProtocol):
    input_directory: Path
    selected_mechanisms: list[str]
    T: list[int]
    t_0: int
    externals: Path

    def __init__(self, vrtool_config: VrtoolConfig) -> DikeSectionImporter:
        if not vrtool_config:
            raise ValueError("VrtoolConfig not provided.")

        self.input_directory = vrtool_config.input_directory
        self.selected_mechanisms = vrtool_config.mechanisms
        self.T = vrtool_config.T
        self.t_0 = vrtool_config.t_0
        self.externals = vrtool_config.externals

    def _import_buildings_list(self, buildings_list: list[Buildings]) -> pd.DataFrame:
        _buildings_data = [
            [_building.distance_from_toe, _building.number_of_buildings]
            for _building in buildings_list
        ]
        return pd.DataFrame(_buildings_data, columns=["distancefromtoe", "cumulative"])

    def _import_geometry(self, profile_points: list[ProfilePoint]) -> pd.DataFrame:
        _importer = GeometryImporter()
        return _importer.import_orm(profile_points)

    def import_orm(self, orm_model: SectionData) -> DikeSection:
        if not orm_model:
            raise ValueError(f"No valid value given for {SectionData.__name__}.")

        _dike_section = DikeSection()
        _dike_section.name = orm_model.section_name
        _dike_section.houses = self._import_buildings_list(orm_model.buildings_list)
        _dike_section.InitialGeometry = self._import_geometry(orm_model.profile_points)
        _dike_section.mechanism_data = {}
        for _mechanism_per_section in orm_model.mechanisms_per_section:
            _dike_section.mechanism_data[_mechanism_per_section.mechanism.name] = ()

        return _dike_section
