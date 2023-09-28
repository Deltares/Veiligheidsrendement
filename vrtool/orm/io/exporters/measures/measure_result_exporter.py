import logging

from vrtool.decision_making.measures.measure_result_collection_protocol import (
    MeasureResultProtocol,
)
from vrtool.orm.io.exporters.orm_exporter_protocol import OrmExporterProtocol
from vrtool.orm.models.measure_per_section import MeasurePerSection
from vrtool.orm.models.measure_result import MeasureResult, MeasureResultParameter
from vrtool.orm.models.measure_result.measure_result_mechanism import (
    MeasureResultMechanism,
)
from vrtool.orm.models.measure_result.measure_result_section import MeasureResultSection
from vrtool.orm.models.mechanism import Mechanism

import pandas as pd


class MeasureResultExporter(OrmExporterProtocol):
    _measure_per_section: MeasurePerSection

    def __init__(self, measure_per_section: MeasurePerSection) -> None:
        self._measure_per_section = measure_per_section

    def _get_parameters_dict(self, measure_result: MeasureResultProtocol) -> dict:
        if isinstance(measure_result, MeasureResultProtocol):
            return dict(
                (k.upper(), v)
                for k, v in measure_result.get_measure_result_parameters().items()
            )
        return {}

    def _export_measure_results_per_time(
        self,
        measure_result: MeasureResultProtocol,
        orm_measure_result: MeasureResult,
        time_value: int,
        time_reliability: pd.Series,
    ):
        MeasureResultSection.create(
            time=time_value,
            beta=time_reliability["Section"],
            cost=measure_result.cost,
            measure_result=orm_measure_result,
        )
        _available_mechanisms = [
            m_idx for m_idx in time_reliability.index if m_idx != "Section"
        ]
        _mr_mechanism = map(
            lambda mechanism_name: dict(
                time=time_value,
                beta=time_reliability[mechanism_name],
                measure_result=orm_measure_result,
                mechanism_per_section=self.get_mechanism_per_section(
                    self._measure_per_section, mechanism_name
                ),
            ),
            _available_mechanisms,
        )
        MeasureResultMechanism.insert_many(_mr_mechanism).execute()

    @staticmethod
    def get_mechanism_per_section(
        measure_per_section: MeasurePerSection, mechanism_name: str
    ):
        return (
            measure_per_section.section.mechanisms_per_section.join(Mechanism)
            .where(Mechanism.name == mechanism_name)
            .get()
        )

    def export_dom(self, measure_result: MeasureResultProtocol) -> None:
        logging.info(
            "STARTED exporting measure id: {}".format(measure_result.measure_id)
        )
        _orm_measure_result = MeasureResult.create(
            measure_per_section=self._measure_per_section,
        )

        # Create the "group" of parameters for this measure.
        def to_params_dict(dict_entry: tuple) -> list[dict]:
            _name, _value = dict_entry
            return dict(
                name=_name, value=float(_value), measure_result=_orm_measure_result
            )

        MeasureResultParameter.insert_many(
            map(
                to_params_dict,
                self._get_parameters_dict(measure_result).items(),
            )
        ).execute()

        # Create (per calculated time) a measure section and as many present mechanisms.
        _measure_reliability = measure_result.section_reliability.SectionReliability
        for time_column in _measure_reliability.columns:
            self._export_measure_results_per_time(
                measure_result,
                _orm_measure_result,
                int(time_column),
                _measure_reliability[time_column],
            )

        logging.info(
            "FINISHED exporting measure id: {}".format(measure_result.measure_id)
        )
