from vrtool.decision_making.measures.measure_result_collection_protocol import (
    MeasureResultProtocol,
)
from vrtool.orm.io.exporters.orm_exporter_protocol import OrmExporterProtocol
import logging

from vrtool.orm.models.measure_per_section import MeasurePerSection
from vrtool.orm.models.measure_result import MeasureResult
from vrtool.orm.models.measure_result_parameter import MeasureResultParameter


class MeasureResultExporter(OrmExporterProtocol):
    _measure_per_section: MeasurePerSection

    def __init__(self, measure_per_section: MeasurePerSection) -> None:
        self._measure_per_section = measure_per_section

    def export_dom(self, measure_result: MeasureResultProtocol) -> None:
        logging.info("STARTED exporting measure id: {}".format(measure_result["id"]))
        _parameters_dict_list = [
            dict(name=m_parameter, value=m_value)
            for m_parameter, m_value in measure_result.get_custom_parameters_dict()
        ]
        for (
            col_name,
            beta_value,
        ) in measure_result.section_reliability.SectionReliability.loc[
            "Section"
        ].iteritems():
            _orm_result = MeasureResult.create(
                beta=beta_value,
                time=int(col_name),
                cost=measure_result.cost,
                measure_per_section=self._measure_per_section,
            )
            _mr_parameters = map(
                lambda p_dict: p_dict | dict(measure_result=_orm_result),
                _parameters_dict_list,
            )
            MeasureResultParameter.insert_many(_mr_parameters).execute()
        logging.info("FINISHED exporting measure id: {}".format(measure_result["id"]))
