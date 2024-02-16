import math
from collections import defaultdict
from typing import Type

from peewee import fn

from vrtool.common.enums.combinable_type_enum import CombinableTypeEnum
from vrtool.common.enums.measure_type_enum import MeasureTypeEnum
from vrtool.common.enums.mechanism_enum import MechanismEnum
from vrtool.defaults.vrtool_config import VrtoolConfig
from vrtool.optimization.measures.measure_as_input_protocol import (
    MeasureAsInputProtocol,
)
from vrtool.optimization.measures.mechanism_per_year import MechanismPerYear
from vrtool.optimization.measures.mechanism_per_year_probability_collection import (
    MechanismPerYearProbabilityCollection,
)
from vrtool.optimization.measures.sg_measure import SgMeasure
from vrtool.optimization.measures.sh_measure import ShMeasure
from vrtool.orm.io.importers.orm_importer_protocol import OrmImporterProtocol
from vrtool.orm.models.measure import Measure as OrmMeasure
from vrtool.orm.models.measure_result.measure_result import (
    MeasureResult as OrmMeasureResult,
)
from vrtool.orm.models.measure_result.measure_result_parameter import (
    MeasureResultParameter as OrmMeasureResultParameter,
)
from vrtool.orm.models.mechanism_per_section import MechanismPerSection
from vrtool.probabilistic_tools.probabilistic_functions import beta_to_pf


class OptimizationMeasureResultImporter(OrmImporterProtocol):

    discount_rate: float
    unit_costs: dict
    investment_year: int

    def __init__(self, vrtool_config: VrtoolConfig, investment_year: int) -> None:
        if not vrtool_config:
            raise ValueError("VrtoolConfig not provided.")

        self.discount_rate = vrtool_config.discount_rate
        self.unit_costs = vrtool_config.unit_costs
        self.investment_year = investment_year

    @staticmethod
    def valid_parameter(measure_result: OrmMeasureResult, parameter_name: str) -> bool:
        """
        Verifies whether the given parameter name exists and is within the expected values
         as a `MeasureResultParameter` for the given `MeasureResult`.

        Args:
            measure_result (MeasureResult): The `MeasureResult` containing a list of parameters.
            parameter_name (str): The parameter name which should be in the `MeasureResult`.

        Returns:
            bool: Parameter is a valid value of the `MeasureResult`.
        """
        _parameter_value = measure_result.get_parameter_value(parameter_name)
        if math.isnan(_parameter_value):
            return False
        return any(math.isclose(_parameter_value, x) for x in [0, -999])

    def _get_mechanism_year_collection(
        self,
        mechanism_measure_results: list[MechanismPerSection],
        allowed_mechanisms: list[MechanismEnum],
    ) -> MechanismPerYearProbabilityCollection:

        _mech_collection = MechanismPerYearProbabilityCollection([])
        for _mech_result in mechanism_measure_results:
            _mech_enum = MechanismEnum.get_enum(
                _mech_result.mechanism_per_section.mechanism.name
            )
            if _mech_enum not in allowed_mechanisms:
                continue

            _mech_per_year = MechanismPerYear(
                _mech_enum,
                year=_mech_result.time,
                probability=beta_to_pf(_mech_result.beta),
            )
            _mech_collection.probabilities.append(_mech_per_year)

        return _mech_collection

    def _create_measure(
        self,
        measure_result: OrmMeasureResult,
        measure_as_input_type: Type[MeasureAsInputProtocol],
    ) -> list[MeasureAsInputProtocol]:
        _mech_year_coll = self._get_mechanism_year_collection(
            measure_result.measure_result_mechanisms,
            measure_as_input_type.get_allowed_mechanisms(),
        )
        _measure_concrete_params = measure_as_input_type.get_concrete_parameters()
        _measures_dict = []
        for _section_result in measure_result.sections_measure_result:
            _time = _section_result.time
            _cost = _section_result.cost
            _measures_dict.append(
                dict(
                    measure_type=MeasureTypeEnum.get_enum(
                        measure_result.measure_per_section.measure.measure_type.name
                    ),
                    combine_type=CombinableTypeEnum.get_enum(
                        measure_result.measure_per_section.measure.combinable_type.name
                    ),
                    cost=_cost,
                    year=_time,
                    discount_rate=self.discount_rate,
                    mechanism_year_collection=_mech_year_coll,
                )
                | {
                    _param: measure_result.get_parameter_value(_param)
                    for _param in _measure_concrete_params
                }
            )

        return list(map(lambda x: measure_as_input_type(**x), _measures_dict))

    def import_orm(self, orm_model: OrmMeasureResult) -> list[MeasureAsInputProtocol]:

        if not orm_model:
            raise ValueError(f"No valid value given for {OrmMeasure.__name__}.")

        _imported_measures = []
        if self.valid_parameter(orm_model, "dberm"):
            _imported_measures.extend(self._create_measure(orm_model, ShMeasure))

        if self.valid_parameter(orm_model, "dcrest"):
            _imported_measures.extend(
                self._create_measure(
                    orm_model,
                    SgMeasure,
                )
            )

        return _imported_measures
