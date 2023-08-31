from vrtool.decision_making.measures.measure_protocol import MeasureProtocol
from vrtool.orm.io.exporters.orm_exporter_protocol import OrmExporterProtocol
from vrtool.orm.models.measure_per_section import MeasurePerSection
from vrtool.orm.models.measure_result import MeasureResult


class SimpleMeasureExporter(OrmExporterProtocol):
    _measure_per_section: MeasurePerSection

    def __init__(self, measure_per_section: MeasurePerSection) -> None:
        self._measure_per_section = measure_per_section

    def export_dom(self, dom_model: MeasureProtocol) -> None:
        _cost = dom_model.measures["Cost"]
        _section_reliability = dom_model.measures["Reliability"]
        _reliabilities_to_export = _section_reliability.SectionReliability.loc[
            "Section"
        ]

        _measure_results = []
        for year in _reliabilities_to_export.index:
            _measure_results.append(
                {
                    "time": year,
                    "measure_per_section": self._measure_per_section,
                    "beta": _reliabilities_to_export[year],
                    "cost": _cost,
                }
            )

        MeasureResult.insert_many(_measure_results).execute()
