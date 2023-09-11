from vrtool.decision_making.measures.measure_result_collection_protocol import (
    MeasureResultProtocol,
)
from vrtool.decision_making.measures.standard_measures.revetment_measure.revetment_measure_section_reliability import (
    RevetmentMeasureSectionReliability,
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

    def _get_parameters_dict(self, measure_result: MeasureResultProtocol) -> dict:
        if isinstance(measure_result, RevetmentMeasureSectionReliability):
            return dict(
                beta_target=measure_result.beta_target,
                transition_level=measure_result.transition_level,
            )
        return dict()

    def export_dom(self, measure_result: MeasureResultProtocol) -> None:
        logging.info(
            "STARTED exporting measure id: {}".format(measure_result.measure_id)
        )
        _parameters_dict_list = [
            dict(name=m_parameter.upper(), value=m_value)
            for m_parameter, m_value in self._get_parameters_dict(
                measure_result
            ).items()
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
        logging.info(
            "FINISHED exporting measure id: {}".format(measure_result.measure_id)
        )
