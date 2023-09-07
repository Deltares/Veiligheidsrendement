import logging

from peewee import fn

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

    def _get_mechanism_per_section(self, mechanism_name: str) -> MechanismPerSection:
        # peewee 'fn' allows us to add query operators. Unfortunately it does not include 'strip'.
        _mechanism = Mechanism.get_or_none(
            fn.Upper(Mechanism.name) == mechanism_name.upper().strip()
        )

        if not _mechanism:
            raise ValueError("No mechanism found for {}.".format(mechanism_name))

        return MechanismPerSection.get_or_none(
            (MechanismPerSection.section == self._section_data)
            & (MechanismPerSection.mechanism == _mechanism)
        )

    def export_dom(self, section_reliability: SectionReliability) -> None:
        logging.info("STARTED exporting Mechanism's reliability (Beta) over time.")
        _added_assessments = []
        _section_reliability = section_reliability.SectionReliability

        for row_idx, mechanism_row in (
            _section_reliability.loc[_section_reliability.index != "Section"]
        ).iterrows():
            logging.info(f"Exporting reliability for mechanism: '{row_idx}'.")
            _mechanism_per_section = self._get_mechanism_per_section(row_idx)

            for time_idx, beta_value in enumerate(mechanism_row):
                _added_assessments.append(
                    AssessmentMechanismResult.create(
                        beta=beta_value,
                        time=int(mechanism_row.index[time_idx]),
                        mechanism_per_section=_mechanism_per_section,
                    )
                )

        logging.info("FINISHED exporting Mechanism's reliability (Beta) over time.")
