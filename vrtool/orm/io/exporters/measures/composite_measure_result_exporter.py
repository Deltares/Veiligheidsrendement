from vrtool.decision_making.measures.measure_protocol import CompositeMeasureProtocol
from vrtool.orm.io.exporters.orm_exporter_protocol import OrmExporterProtocol
from vrtool.orm.models.measure_per_section import MeasurePerSection
from vrtool.orm.models.measure_result import MeasureResult

import logging

from vrtool.orm.models.measure_result_parameter import MeasureResultParameter

_composite_parameters = ["dcrest", "dberm"]


class CompositeMeasureResultExporter(OrmExporterProtocol):
    _measure_per_section: MeasurePerSection

    def __init__(self, measure_per_section: MeasurePerSection) -> None:
        self._measure_per_section = measure_per_section

    def export_dom(self, composite_measure: CompositeMeasureProtocol) -> None:
        logging.info(
            "STARTED exporting measure's results for {}".format(type(composite_measure))
        )
        for _measure in composite_measure.measures:
            _available_parameters = list(
                filter(lambda x: x in _measure, _composite_parameters)
            )
            for col_name, beta_value in _measure["Reliability"]:
                _measure_result = MeasureResult.create(
                    beta=beta_value,
                    time=int(col_name),
                    cost=_measure["Cost"],
                    measure_per_section=self._measure_per_section,
                )
                for _parameter in _available_parameters:
                    MeasureResultParameter.create(
                        name=_parameter.upper(),
                        value=_measure[_parameter],
                        measure_result=_measure_result,
                    )
        logging.info("FINISHED exporting measure's results.")
