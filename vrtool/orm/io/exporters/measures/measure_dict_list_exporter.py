import logging

from vrtool.orm.io.exporters.orm_exporter_protocol import OrmExporterProtocol
from vrtool.orm.models.measure_per_section import MeasurePerSection
from vrtool.orm.models.measure_result import MeasureResult, MeasureResultParameter

_supported_parameters = ["dcrest", "dberm"]


class MeasureDictListExporter(OrmExporterProtocol):
    _measure_per_section: MeasurePerSection

    def __init__(self, measure_per_section: MeasurePerSection) -> None:
        self._measure_per_section = measure_per_section

    def _export_section_measure_result(self, _measure: dict) -> None:
        _available_parameters = list(
            filter(lambda x: x in _measure, _supported_parameters)
        )
        _measure_cost = _measure["Cost"]
        for col_name, beta_value in (
            _measure["Reliability"].SectionReliability.loc["Section"].iteritems()
        ):
            _measure_result = MeasureResult.create(
                beta=beta_value,
                time=int(col_name),
                cost=_measure_cost,
                measure_per_section=self._measure_per_section,
            )
            _mr_parameters = map(
                lambda x: dict(
                    name=x.upper(),
                    value=_measure[x],
                    measure_result=_measure_result,
                ),
                _available_parameters,
            )
            MeasureResultParameter.insert_many(_mr_parameters).execute()

    def export_dom(self, measure_dict_list: list) -> None:
        # TODO: Potentially this could be done in SimpleMeasureExporter and here only iterate over the measures.
        logging.info("STARTED exporting measure's results list.")
        for _measure in measure_dict_list:
            logging.debug("STARTED exporting measure id: {}".format(_measure["id"]))
            self._export_section_measure_result(_measure)
            logging.debug("FINISHED exporting measure id: {}".format(_measure["id"]))

        logging.info("FINISHED exporting measure's results.")
