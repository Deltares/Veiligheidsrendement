import pandas as pd

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

    def export_dom(self, measure_result: MeasureResultProtocol) -> None:
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
        _measure_result_mechanisms_list_dict = []
        _measure_result_section_list_dict = []
        for time_column in _measure_reliability.columns:
            _time = int(time_column)
            _time_reliability = _measure_reliability[time_column]
            _measure_result_section_list_dict.append(
                self._get_measure_result_section_dict(
                    measure_result,
                    _orm_measure_result,
                    _time,
                    _time_reliability,
                )
            )
            _measure_result_mechanisms_list_dict.extend(
                self._get_measure_result_mechanism_list_dict(
                    _orm_measure_result, _time, _time_reliability
                )
            )

        MeasureResultSection.insert_many(_measure_result_section_list_dict).execute()
        MeasureResultMechanism.insert_many(
            _measure_result_mechanisms_list_dict
        ).execute()
