from __future__ import annotations

import logging
from collections import defaultdict
from math import isclose
from pathlib import Path

import pandas as pd

from vrtool.common.enums.computation_type_enum import ComputationTypeEnum
from vrtool.common.enums.mechanism_enum import MechanismEnum
from vrtool.defaults.vrtool_config import VrtoolConfig
from vrtool.flood_defence_system.dike_section import DikeSection
from vrtool.flood_defence_system.mechanism_reliability_collection import (
    MechanismReliabilityCollection,
)
from vrtool.flood_defence_system.section_reliability import SectionReliability
from vrtool.orm.io.importers.geometry_importer import GeometryImporter
from vrtool.orm.io.importers.mechanism_reliability_collection_importer import (
    MechanismReliabilityCollectionImporter,
)
from vrtool.orm.io.importers.orm_importer_protocol import OrmImporterProtocol
from vrtool.orm.io.importers.water_level_importer import WaterLevelImporter
from vrtool.orm.io.validators.section_data_validator import SectionDataValidator
from vrtool.orm.models.assessment_mechanism_result import AssessmentMechanismResult
from vrtool.orm.models.assessment_section_result import AssessmentSectionResult
from vrtool.orm.models.buildings import Buildings
from vrtool.orm.models.mechanism_per_section import MechanismPerSection
from vrtool.orm.models.section_data import SectionData


class DikeSectionImporter(OrmImporterProtocol):
    input_directory: Path
    selected_mechanisms: list[MechanismEnum]
    computation_years: list[int]
    t_0: int
    externals: Path

    def __init__(self, vrtool_config: VrtoolConfig) -> None:
        if not vrtool_config:
            raise ValueError("VrtoolConfig not provided.")

        self.input_directory = vrtool_config.input_directory
        self.selected_mechanisms = vrtool_config.mechanisms
        self.computation_years = vrtool_config.T
        self.t_0 = vrtool_config.t_0
        self.externals = vrtool_config.externals
        self._config = vrtool_config

    @staticmethod
    def import_assessment_reliability_df(section_data: SectionData) -> pd.DataFrame:
        """
        Imports the assessment reliability data related to a section as a `pd.DataFrame`.

        Args:
            section_data (SectionData): Section Data with an already saved initial assessment.

        Returns:
            pd.DataFrame: Dataframe containing information of section and mechanisms assessments.
        """
        _columns = []
        _section_reliability_dict = defaultdict(list)
        for _asr in section_data.assessment_section_results.order_by(
            AssessmentSectionResult.time.asc()
        ):
            _columns.append(str(_asr.time))
            _section_reliability_dict["Section"].append(_asr.beta)
            for _amr in (
                AssessmentMechanismResult.select()
                .join(MechanismPerSection)
                .where(
                    (AssessmentMechanismResult.time == _asr.time)
                    & (
                        AssessmentMechanismResult.mechanism_per_section.section
                        == section_data
                    ),
                )
            ):
                _mech_name = MechanismEnum.get_enum(
                    _amr.mechanism_per_section.mechanism.name
                ).name
                _section_reliability_dict[_mech_name].append(_amr.beta)
        return pd.DataFrame.from_dict(
            _section_reliability_dict, columns=_columns, orient="index"
        )

    def _import_buildings_list(self, buildings_list: list[Buildings]) -> pd.DataFrame:
        _buildings_data = [
            [_building.distance_from_toe, _building.number_of_buildings]
            for _building in buildings_list
        ]
        return pd.DataFrame(
            _buildings_data, columns=["distancefromtoe", "cumulative"]
        ).set_index("distancefromtoe")

    def _import_geometry(self, section_data: SectionData) -> pd.DataFrame:
        _importer = GeometryImporter()
        return _importer.import_orm(section_data)

    def _get_mechanism_reliability_collection_list(
        self, section_data: SectionData
    ) -> list[MechanismReliabilityCollection]:
        _importer = MechanismReliabilityCollectionImporter(self._config)
        _mechanism_data = []
        for _mechanism_per_section in section_data.mechanisms_per_section:
            if not any(_mechanism_per_section.computation_scenarios):
                logging.error(
                    "No computation scenarios available for Section %s - Mechanism: %s",
                    _mechanism_per_section.section.section_name,
                    _mechanism_per_section.mechanism.name,
                )
            elif (
                MechanismEnum.get_enum(_mechanism_per_section.mechanism.name)
                not in self.selected_mechanisms
            ):
                logging.debug(
                    "Skipping import of mechanism %s, as it is not selected in the configuration.",
                    _mechanism_per_section.mechanism.name,
                )

            else:
                _mechanism_data.append(_importer.import_orm(_mechanism_per_section))
        return _mechanism_data

    def _get_mechanism_data(
        self, section_data: SectionData
    ) -> dict[MechanismEnum, list[tuple[str, ComputationTypeEnum]]]:
        _mechanism_data = {}
        for _mechanism_per_section in section_data.mechanisms_per_section:
            _available_cs = []
            for _cs in _mechanism_per_section.computation_scenarios:
                _available_cs.append(
                    (
                        _cs.scenario_name,
                        ComputationTypeEnum.get_enum(_cs.computation_type.name),
                    )
                )
            _mechanism = MechanismEnum.get_enum(_mechanism_per_section.mechanism.name)
            _mechanism_data[_mechanism] = _available_cs
        return _mechanism_data

    def _get_section_reliability(
        self,
        section_data: SectionData,
    ) -> SectionReliability:
        _section_reliability = SectionReliability()

        _section_reliability.load = WaterLevelImporter(gridpoints=1000).import_orm(
            section_data
        )

        _mechanism_collection = self._get_mechanism_reliability_collection_list(
            section_data
        )

        _imported_initial_assessment = self.import_assessment_reliability_df(
            section_data
        )

        for _mechanism_data in _mechanism_collection:
            if _mechanism_data.mechanism.name in _imported_initial_assessment.index:
                _mech_name = MechanismEnum.get_enum(_mechanism_data.mechanism.name).name
                for _reliability_t, _beta in _imported_initial_assessment.loc[
                    _mech_name
                ].items():
                    _mechanism_data.Reliability[_reliability_t].Beta = _beta
            _section_reliability.failure_mechanisms.add_failure_mechanism_reliability_collection(
                _mechanism_data
            )

        if _imported_initial_assessment.empty:
            logging.debug(
                "No initial section -  mechanism (reliability) assessment was found."
            )
        else:
            _section_reliability.SectionReliability = _imported_initial_assessment

        return _section_reliability

    def import_orm(self, orm_model: SectionData) -> DikeSection:
        _report = SectionDataValidator().validate(orm_model)
        if any(_report.errors):
            # Raise exception with the first error found.
            raise ValueError(_report.errors[0].error_message)

        _dike_section = DikeSection(
            name=orm_model.section_name,
            houses=self._import_buildings_list(orm_model.buildings_list),
            InitialGeometry=self._import_geometry(orm_model),
            # TODO: Not entirely sure mechanism_data is correctly set. Technically should not be needed anymore.
            mechanism_data=self._get_mechanism_data(orm_model),
            section_reliability=self._get_section_reliability(orm_model),
            Length=orm_model.section_length,
            crest_height=orm_model.crest_height,
            cover_layer_thickness=orm_model.cover_layer_thickness,
            pleistocene_level=orm_model.pleistocene_level,
            flood_damage=orm_model.get_flood_damage_value(),
            sensitive_fraction_piping=orm_model.sensitive_fraction_piping,
            sensitive_fraction_stability_inner=orm_model.sensitive_fraction_stability_inner,
        )

        return _dike_section
