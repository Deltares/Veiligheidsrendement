from typing import Any
from vrtool.flood_defence_system.dike_section import DikeSection
from vrtool.flood_defence_system.section_reliability import SectionReliability
from vrtool.orm.io.exporters.orm_exporter_protocol import OrmExporterProtocol
from vrtool.orm.models.assessment_section_results import AssessmentSectionResults
from vrtool.orm.models.dike_traject_info import DikeTrajectInfo
from vrtool.orm.models.orm_base_model import OrmBaseModel
from vrtool.orm.models.section_data import SectionData


class SectionReliabilityExporter(OrmExporterProtocol):
    _section_data: SectionData

    def __init__(self, section_data: SectionData) -> None:
        self._section_data = section_data

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
                SectionData.dike_traject.traject_name
                == dike_section.TrajectInfo.traject_name
            )
            .get_or_none()
        )

    def export_dom(
        self, dom_model: SectionReliability
    ) -> list[AssessmentSectionResults]:
        for reliability_df in dom_model.SectionReliability:
            AssessmentSectionResults.create(
                beta=...,
                time=...,
            )
