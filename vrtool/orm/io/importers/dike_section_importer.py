from __future__ import annotations

from pathlib import Path

import pandas as pd

from vrtool.defaults.vrtool_config import VrtoolConfig
from vrtool.flood_defence_system.dike_section import DikeSection
from vrtool.orm.io.importers.geometry_importer import GeometryImporter
from vrtool.orm.io.importers.mechanism_reliability_collection_importer import MechanismReliabilityCollectionImporter
from vrtool.orm.io.importers.orm_importer_protocol import OrmImporterProtocol
from vrtool.orm.models.buildings import Buildings
from vrtool.orm.models.section_data import SectionData
from vrtool.orm.models.mechanism_per_section import MechanismPerSection


class DikeSectionImporter(OrmImporterProtocol):
    input_directory: Path
    selected_mechanisms: list[str]
    computation_years: list[int]
    t_0: int
    externals: Path

    def __init__(self, vrtool_config: VrtoolConfig) -> DikeSectionImporter:
        if not vrtool_config:
            raise ValueError("VrtoolConfig not provided.")

        self.input_directory = vrtool_config.input_directory
        self.selected_mechanisms = vrtool_config.mechanisms
        self.computation_years = vrtool_config.T
        self.t_0 = vrtool_config.t_0
        self.externals = vrtool_config.externals
        self._config = vrtool_config

    def _import_buildings_list(self, buildings_list: list[Buildings]) -> pd.DataFrame:
        _buildings_data = [
            [_building.distance_from_toe, _building.number_of_buildings]
            for _building in buildings_list
        ]
        return pd.DataFrame(_buildings_data, columns=["distancefromtoe", "cumulative"])

    def _import_geometry(self, section_data: SectionData) -> pd.DataFrame:
        _importer = GeometryImporter()
        return _importer.import_orm(section_data)

    def _get_mechanism_data(self, section_data: SectionData) -> dict[str, MechanismPerSection]:
        _importer = MechanismReliabilityCollectionImporter(self._config)
        _mechanism_data = {}
        for _mechanism_per_section in section_data.mechanisms_per_section:
            _mechanism_data[_mechanism_per_section.mechanism.name] = _importer.import_orm(_mechanism_per_section)
        return _mechanism_data

    def import_orm(self, orm_model: SectionData) -> DikeSection:
        if not orm_model:
            raise ValueError(f"No valid value given for {SectionData.__name__}.")

        _dike_section = DikeSection()
        _dike_section.name = orm_model.section_name
        _dike_section.houses = self._import_buildings_list(orm_model.buildings_list)
        _dike_section.InitialGeometry = self._import_geometry(orm_model)
        _dike_section.mechanism_data = self._get_mechanism_data(orm_model)        
        return _dike_section

