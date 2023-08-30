from vrtool.decision_making.measures.measure_protocol import MeasureProtocol
from vrtool.orm.io.exporters.orm_exporter_protocol import OrmExporterProtocol
from vrtool.orm.models.measure_per_section import MeasurePerSection


class SimpleMeasureExporter(OrmExporterProtocol):
    _measure_per_section: MeasurePerSection

    def __init__(self, measure_per_section: MeasurePerSection) -> None:
        self._measure_per_section = measure_per_section

    def export_dom(self, dom_model: MeasureProtocol) -> None:
        # Retrieve self.measures from the measure protocol --> Pull member up to the measure protocol
        # Retrieve the following properties:
        # - self.measures["Cost"]
        # - self.measures["Reliability"]
        # --> Results in a SectionReliability --> which has a property sectionReliability
        # (which is a pandas dataframe with the beta_time)

        pass
