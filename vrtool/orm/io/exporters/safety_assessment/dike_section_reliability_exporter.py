import logging

from vrtool.flood_defence_system.dike_section import DikeSection
from vrtool.orm.io.exporters.orm_exporter_protocol import OrmExporterProtocol
from vrtool.orm.io.exporters.safety_assessment.mechanism_reliability_collection_exporter import (
    MechanismReliabilityCollectionExporter,
)
from vrtool.orm.io.exporters.safety_assessment.section_reliability_exporter import (
    SectionReliabilityExporter,
)
from vrtool.orm.models.dike_traject_info import DikeTrajectInfo
from vrtool.orm.models.section_data import SectionData


class DikeSectionReliabilityExporter(OrmExporterProtocol):
    @staticmethod
    def get_related_section_data(dike_section: DikeSection) -> SectionData:
        """
        Retrieves the database's mapped dike section as an ORM object.

        Args:
            dike_section (DikeSection): DOM dike section saved in the database.

        Returns:
            SectionData: ORM instance representing the given `DikeSection` or `None` when no match was found.
        """
        return (
            SectionData.select()
            .join(DikeTrajectInfo)
            .where(
                (DikeTrajectInfo.traject_name == dike_section.TrajectInfo.traject_name)
                & (SectionData.section_name == dike_section.name)
            )
            .get_or_none()
        )

    def export_dom(self, dom_model: DikeSection) -> None:
        logging.debug(
            "STARTED exporting Dike Section's initial assessment reliability."
        )
        _section_data = self.get_related_section_data(dom_model)
        MechanismReliabilityCollectionExporter(_section_data).export_dom(
            dom_model.section_reliability
        )
        SectionReliabilityExporter(_section_data).export_dom(
            dom_model.section_reliability
        )
        logging.debug(
            "FINISHED exporting Dike Section's initial assessment reliability."
        )
