from __future__ import annotations

from pathlib import Path

import pandas as pd
from vrtool.common.hydraulic_loads.load_input import LoadInput

from vrtool.defaults.vrtool_config import VrtoolConfig
from vrtool.flood_defence_system.dike_section import DikeSection
from vrtool.flood_defence_system.mechanism_reliability_collection import MechanismReliabilityCollection
from vrtool.flood_defence_system.section_reliability import SectionReliability
from vrtool.orm.io.importers.geometry_importer import GeometryImporter
from vrtool.orm.io.importers.mechanism_reliability_collection_importer import MechanismReliabilityCollectionImporter
from vrtool.orm.io.importers.orm_importer_protocol import OrmImporterProtocol
from vrtool.orm.io.importers.water_level_importer import WaterLevelImporter
from vrtool.orm.models.buildings import Buildings
from vrtool.orm.models.section_data import SectionData
from vrtool.orm.models.mechanism_per_section import MechanismPerSection
import logging

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

    def _get_mechanism_data(self, section_data: SectionData) -> dict[str, MechanismReliabilityCollection]:
        _importer = MechanismReliabilityCollectionImporter(self._config)
        _mechanism_data = {}
        for _mechanism_per_section in section_data.mechanisms_per_section:
            if not any(_mechanism_per_section.computation_scenarios):
                logging.error("No computation scenarios available for Section {} - Mechanism: {}".format(_mechanism_per_section.section.section_name, _mechanism_per_section.mechanism.name))
                _mechanism_data[_mechanism_per_section.mechanism.name] = ()
            else:
                _mechanism_data[_mechanism_per_section.mechanism.name] = _importer.import_orm(_mechanism_per_section)
        return _mechanism_data

    def _get_section_reliability(self, section_data: SectionData, mechanism_collection: dict[str, MechanismReliabilityCollection]) -> SectionReliability:
        _section_reliability = SectionReliability()
        _section_reliability.load =  WaterLevelImporter(gridpoints=1000).import_orm(section_data)

        for _mechanism_data in mechanism_collection.values():
            _section_reliability.failure_mechanisms.add_failure_mechanism_reliability_collection(_mechanism_data)
        
        return _section_reliability

    def import_orm(self, orm_model: SectionData) -> DikeSection:
        if not orm_model:
            raise ValueError(f"No valid value given for {SectionData.__name__}.")

        _dike_section = DikeSection()
        _dike_section.name = orm_model.section_name
        _dike_section.houses = self._import_buildings_list(orm_model.buildings_list)
        _dike_section.InitialGeometry = self._import_geometry(orm_model)
        # TODO: Not entirely sure mechanism_data is correctly set. Technically should not be needed anymore.
        _dike_section.mechanism_data = self._get_mechanism_data(orm_model) 
        _dike_section.section_reliability = self._get_section_reliability(orm_model, _dike_section.mechanism_data)

        return _dike_section

