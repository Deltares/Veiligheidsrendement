import pandas as pd
from peewee import chunked

from vrtool.common.enums.mechanism_enum import MechanismEnum
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
from vrtool.orm.models.mechanism_per_section import MechanismPerSection


class MeasureResultExporter(OrmExporterProtocol):
    _measure_per_section: MeasurePerSection

    def __init__(self, measure_per_section: MeasurePerSection) -> None:
        self._measure_per_section = measure_per_section

    @staticmethod
    def get_mechanism_per_section(
        measure_per_section: MeasurePerSection, mechanism: MechanismEnum
    ) -> MechanismPerSection:
        """
        Retrieves the associated `MechanismPerSection` for a given
        `SectionData` and a given `mechanism_name`.

        Args:
            measure_per_section (MeasurePerSection): Instance used
             to derive de `SectionData` row.
            mechanism (MechanismEnum: Desired `Mechanism` related entry.

        Returns:
            MechanismPerSection: Instance connected to the provided
             `MeasurePerSection`.
        """
        return (
            measure_per_section.section.mechanisms_per_section.join(Mechanism)
            .where(Mechanism.name << [mechanism.name, mechanism.legacy_name])
            .get()
        )

    def _get_parameters_dict(self, measure_result: MeasureResultProtocol) -> dict:
        if isinstance(measure_result, MeasureResultProtocol):
            return dict(
                (k.upper(), v)
                for k, v in measure_result.get_measure_result_parameters().items()
            )
        return {}

    def _get_measure_result_section_dict(
        self,
        measure_result: MeasureResultProtocol,
        orm_measure_result: MeasureResult,
        time_value: int,
        time_reliability: pd.Series,
    ) -> dict:
        return dict(
            time=time_value,
            beta=time_reliability["Section"],
            cost=measure_result.cost,
            measure_result=orm_measure_result,
        )

    def _get_measure_result_mechanism_list_dict(
        self,
        orm_measure_result: MeasureResult,
        time_value: int,
        time_reliability: pd.Series,
    ) -> list[dict]:
        _available_mechanisms = [
            MechanismEnum.get_enum(m_idx)
            for m_idx in time_reliability.index
            if m_idx != "Section"
        ]
        return list(
            map(
                lambda mechanism: dict(
                    time=time_value,
                    beta=time_reliability[mechanism.name],
                    measure_result=orm_measure_result,
                    mechanism_per_section=self.get_mechanism_per_section(
                        self._measure_per_section, mechanism
                    ),
                ),
                _available_mechanisms,
            )
        )

    def _create_measure_results(
        self, measure_result_collection: list[MeasureResultProtocol]
    ) -> list[MeasureResult]:
        _rm_dict = dict(measure_per_section=self._measure_per_section)
        MeasureResult.insert_many([_rm_dict] * len(measure_result_collection)).execute()
        return list(
            MeasureResult.select().where(
                MeasureResult.measure_per_section == self._measure_per_section
            )
        )

    def export_dom(self, dom_model: list[MeasureResultProtocol]) -> None:
        _orm_measure_result_list: list[MeasureResult] = self._create_measure_results(
            dom_model
        )

        # Create the "group" of parameters for this measure.
        def to_params_dict(dict_entry: tuple, _mr_model: MeasureResult) -> list[dict]:
            _name, _value = dict_entry
            return dict(name=_name, value=float(_value), measure_result=_mr_model)

        _mrp = []
        _mrs = []
        _mrm = []
        for i, _result in enumerate(dom_model):
            _mrp.extend(
                [
                    to_params_dict(param, _orm_measure_result_list[i])
                    for param in self._get_parameters_dict(_result).items()
                ]
            )

            _measure_reliability = _result.section_reliability.SectionReliability
            for time_column in _measure_reliability.columns:
                _time = int(time_column)
                _time_reliability = _measure_reliability[time_column]
                _mrs.append(
                    self._get_measure_result_section_dict(
                        _result,
                        _orm_measure_result_list[i],
                        _time,
                        _time_reliability,
                    )
                )
                _mrm.extend(
                    self._get_measure_result_mechanism_list_dict(
                        _orm_measure_result_list[i], _time, _time_reliability
                    )
                )

        for _mrp_chunk in chunked(_mrp, 200):
            MeasureResultParameter.insert_many(_mrp_chunk).execute()
        for _mrs_chunk in chunked(_mrs, 200):
            MeasureResultSection.insert_many(_mrs_chunk).execute()
        for _mrm_chunk in chunked(_mrm, 200):
            MeasureResultMechanism.insert_many(_mrm_chunk).execute()
