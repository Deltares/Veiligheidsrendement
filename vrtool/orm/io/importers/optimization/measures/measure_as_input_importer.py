from copy import deepcopy
from typing import Any

from vrtool.common.enums.combinable_type_enum import CombinableTypeEnum
from vrtool.common.enums.measure_type_enum import MeasureTypeEnum
from vrtool.common.enums.mechanism_enum import MechanismEnum
from vrtool.optimization.measures.measure_as_input_protocol import (
    MeasureAsInputProtocol,
)
from vrtool.optimization.measures.mechanism_per_year import MechanismPerYear
from vrtool.optimization.measures.mechanism_per_year_probability_collection import (
    MechanismPerYearProbabilityCollection,
)
from vrtool.orm.io.importers.optimization.measures.measure_as_input_importer_data import (
    MeasureAsInputImporterData,
)
from vrtool.orm.models.measure_result.measure_result_section import (
    MeasureResultSection as OrmMeasureResultSection,
)
from vrtool.probabilistic_tools.probabilistic_functions import beta_to_pf


class MeasureAsInputImporter:
    """
    This importer converts a `MeasureResult` from the orm into one or more
    instances of a concrete type of `MeasureAsInputProtocol`
    (`ShMeasure`, `SgMeasure` or `ShSgMeasure).

    To improve maintainability `MeasureAsInputImporterData` was introduced
    so that all three concrete types can be created within this generic importer.
    """

    def __init__(
        self, measure_as_input_importer_data: MeasureAsInputImporterData
    ) -> None:
        self._measure_result = measure_as_input_importer_data.measure_result
        self._investment_years = measure_as_input_importer_data.investment_years
        self._measure_as_input_type = (
            measure_as_input_importer_data.measure_as_input_type
        )
        self._mech_year_coll = self._get_mechanism_year_collection()
        self._discount_rate = measure_as_input_importer_data.discount_rate
        self._concrete_parameters = measure_as_input_importer_data.concrete_parameters

    def _get_mechanism_year_collection(self) -> MechanismPerYearProbabilityCollection:
        _mech_collection = MechanismPerYearProbabilityCollection([])
        _allowed_mechanisms = self._measure_as_input_type.get_allowed_mechanisms()
        for _mech_result in self._measure_result.measure_result_mechanisms:
            _mech_enum = MechanismEnum.get_enum(
                _mech_result.mechanism_per_section.mechanism.name
            )
            if _mech_enum not in _allowed_mechanisms:
                continue

            _mech_per_year = MechanismPerYear(
                _mech_enum,
                year=_mech_result.time,
                probability=beta_to_pf(_mech_result.beta),
            )
            _mech_collection.probabilities.append(_mech_per_year)

        return _mech_collection

    def _get_measure_as_input_dictionary(
        self, section_cost: float, investment_year: int
    ) -> dict[str, Any]:
        return dict(
            measure_result_id=self._measure_result.id,
            measure_type=MeasureTypeEnum.get_enum(
                self._measure_result.measure_per_section.measure.measure_type.name
            ),
            combine_type=CombinableTypeEnum.get_enum(
                self._measure_result.measure_per_section.measure.combinable_type.name
            ),
            cost=section_cost,
            base_cost=0.0,
            year=investment_year,
            discount_rate=self._discount_rate,
            # Important!
            # The `MechanismYearCollection` needs to be deep copied. Otherwise
            # its data might be altered affecting all the measures with the same
            # reference.
            mechanism_year_collection=deepcopy(self._mech_year_coll),
        ) | {
            _parameter_name: self._measure_result.get_parameter_value(_parameter_name)
            for _parameter_name in self._concrete_parameters
        }

    def import_measure_as_input_collection(self) -> list[MeasureAsInputProtocol]:
        """
        Creates a list of concrete `MeasureAsInputProtocol` based on all the
        investment years and `MechanismPerYearProbabilityCollection`.

        Returns:
            list[MeasureAsInputProtocol]: list of concrete `MeasureAsInputProtocol`.
        """
        _measures_dicts = []
        for _section_result in self._measure_result.measure_result_section.where(
            OrmMeasureResultSection.time == 0
        ):
            # Create a measure for each investment year (VRTOOL-431).
            _measures_dicts.extend(
                [
                    self._get_measure_as_input_dictionary(
                        _section_result.cost, _investment_year
                    )
                    for _investment_year in self._investment_years
                ]
            )

        return list(
            map(
                lambda x: self._measure_as_input_type(**x),
                _measures_dicts,
            )
        )
