from vrtool.decision_making.measures.measure_protocol import SimpleMeasureProtocol
from vrtool.orm.io.exporters.orm_exporter_protocol import OrmExporterProtocol
from vrtool.orm.models.measure_per_section import MeasurePerSection
from vrtool.orm.models.measure_result import MeasureResult


class SimpleMeasureExporter(OrmExporterProtocol):
    _measure_per_section: MeasurePerSection

    def __init__(self, measure_per_section: MeasurePerSection) -> None:
        self._measure_per_section = measure_per_section

    def export_dom(self, dom_model: SimpleMeasureProtocol) -> None:
        _cost = dom_model.measures["Cost"]
        _section_reliability = dom_model.measures["Reliability"]
        _reliabilities_to_export = _section_reliability.SectionReliability.loc[
            "Section"
        ]

        MeasureResult.insert_many(
            map(
                lambda year: {
                    "time": year,
                    "measure_per_section": self._measure_per_section,
                    "beta": _reliabilities_to_export[year],
                    "cost": _cost,
                },
                _reliabilities_to_export.index,
            )
        ).execute()
