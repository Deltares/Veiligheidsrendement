import logging

from vrtool.decision_making.solutions import Solutions
from vrtool.orm.io.exporters.measures.measure_exporter import MeasureExporter
from vrtool.orm.io.exporters.orm_exporter_protocol import OrmExporterProtocol
from vrtool.orm.models import DikeTrajectInfo, Measure, MeasurePerSection, SectionData


class SolutionsExporter(OrmExporterProtocol):
    @staticmethod
    def get_measure_per_section(
        dike_section_name: str, traject_name: str, measure_id: int
    ) -> MeasurePerSection:
        """
        Gets an instance of a `MeasurePerSection` given the database contains an entry
        where its `SectionData.section_name` and
        `SectionData.dike_traject.traject_name` match the ones provided as argument;
        otherwise gets None.

        Args:
            dike_section_name (str): Value matching a `SectionData.section_name` entry.
            traject_name (str): Value matching a `DikeTraject.traject_name` entry.
            measure_id (int): Id of an existing `Measure` in the database.

        Raises:
            ValueError: When no `Measure` entry was found with the provided
                `measure_id` as `Measure.id`.
            ValueError: When no `SectionData` entry was found with a matching
                `SectionData.section_name` or its
                `SectionData.dike_traject.traject_name` is not the same as the
                provided `traject_name`.

        Returns:
            MeasurePerSection: Found instance with matching values or None
                (`SectionData` and `Measure` exist but no combination in
                `MeasurePerSection` was found).
        """
        _measure = Measure.get_or_none(Measure.id == measure_id)
        if not _measure:
            raise ValueError(f"No 'Measure' was found with id: {measure_id}.")
        _section_data = (
            SectionData.select()
            .join(DikeTrajectInfo)
            .where(
                (SectionData.section_name == dike_section_name)
                & (DikeTrajectInfo.traject_name == traject_name)
            )
            .get_or_none()
        )
        if not _section_data:
            raise ValueError(
                f"No 'SectionData' was found with name: {dike_section_name}, for 'DikeTraject': {traject_name}."
            )
        return (
            MeasurePerSection.select()
            .where(
                (MeasurePerSection.measure == _measure)
                # This ensures we are getting the dike section of the expected dike traject.
                & (MeasurePerSection.section == _section_data)
            )
            .get_or_none()
        )

    def export_dom(self, dom_model: Solutions) -> None:
        _logging_exporting = "exporting solutions for section {}.".format(
            dom_model.section_name
        )
        logging.debug("STARTED {}".format(_logging_exporting))
        for measure in dom_model.measures:
            _measure_per_section = self.get_measure_per_section(
                dom_model.section_name,
                dom_model.config.traject,
                measure.parameters["ID"],
            )
            MeasureExporter(_measure_per_section).export_dom(measure)
        logging.debug(
            "Exported measures for section {}.".format(dom_model.section_name)
        )
