import logging

from vrtool.common.enums.mechanism_enum import MechanismEnum
from vrtool.flood_defence_system.section_reliability import SectionReliability
from vrtool.orm.io.exporters.orm_exporter_protocol import OrmExporterProtocol
from vrtool.orm.models.assessment_mechanism_result import AssessmentMechanismResult
from vrtool.orm.models.mechanism import Mechanism
from vrtool.orm.models.mechanism_per_section import MechanismPerSection
from vrtool.orm.models.section_data import SectionData


class MechanismReliabilityCollectionExporter(OrmExporterProtocol):
    _section_data: SectionData

    def __init__(self, section_data: SectionData) -> None:
        self._section_data = section_data

    def _get_mechanism_per_section(
        self, mechanism: MechanismEnum
    ) -> MechanismPerSection:
        _mech_inst = Mechanism.get_or_none(
            Mechanism.name << [mechanism.name, mechanism.legacy_name]
        )

        if not _mech_inst:
            raise ValueError("No mechanism found for {}.".format(mechanism))

        return MechanismPerSection.get_or_none(
            (MechanismPerSection.section == self._section_data)
            & (MechanismPerSection.mechanism == _mech_inst)
        )

    def export_dom(self, section_reliability: SectionReliability) -> None:
        logging.debug(
            "STARTED exporting Mechanism's reliability (Beta) over time for section {}".format(
                self._section_data.section_name
            )
        )
        _section_reliability = section_reliability.SectionReliability

        for row_idx, mechanism_row in (
            _section_reliability.loc[_section_reliability.index != "Section"]
        ).iterrows():
            _mechanism = MechanismEnum.get_enum(row_idx)
            logging.debug(f"Exporting reliability for mechanism: '{_mechanism}'.")
            _mechanism_per_section = self._get_mechanism_per_section(_mechanism)
            _assessment_list = []
            for time_idx, beta_value in enumerate(mechanism_row):
                _assessment_list.append(
                    dict(
                        beta=beta_value,
                        time=int(mechanism_row.index[time_idx]),
                        mechanism_per_section=_mechanism_per_section,
                    )
                )
            AssessmentMechanismResult.insert_many(_assessment_list).execute()

        logging.debug("FINISHED exporting Mechanism's reliability (Beta) over time.")
