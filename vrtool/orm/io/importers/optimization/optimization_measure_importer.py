from collections import defaultdict
import math
from typing import Type

import pandas as pd
from vrtool.common.enums.combinable_type_enum import CombinableTypeEnum
from vrtool.common.enums.measure_type_enum import MeasureTypeEnum
from vrtool.common.enums.mechanism_enum import MechanismEnum


from vrtool.decision_making.measures.measure_protocol import MeasureProtocol
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

from peewee import fn


class OptimizationMeasureImporter(OrmImporterProtocol):

    discount_rate: float
    unit_costs: dict

    def __init__(self, vrtool_config: VrtoolConfig) -> None:
        if not vrtool_config:
            raise ValueError("VrtoolConfig not provided.")

        self.discount_rate = vrtool_config.discount_rate
        self.unit_costs = vrtool_config.unit_costs

    @staticmethod
    def get_parameter(measure_result: OrmMeasureResult, parameter_name: str) -> float:
        _values = measure_result.measure_result_parameters.where(
            fn.Lower(OrmMeasureResultParameter.name) == parameter_name.lower()
        ).select()
        return _values[0].value if any(_values) else float("nan")

    @staticmethod
    def valid_parameter(measure_result: OrmMeasureResult, parameter_name: str) -> bool:
        _parameter_value = OptimizationMeasureImporter.get_parameter(
            measure_result, parameter_name
        )
        if math.isnan(_parameter_value):
            return False
        return any(math.isclose(_parameter_value, x) for x in [0, -999])

    @staticmethod
    def _get_mechanism_year_collection(
        mechanism_measure_results: list[MechanismPerSection],
        allowed_mechanisms: list[MechanismEnum],
    ) -> dict[int, MechanismPerYearProbabilityCollection]:

        _prob_dictionary = defaultdict(list)
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
            _prob_dictionary[_mech_per_year.year].append(_mech_per_year)

        return {
            _year: MechanismPerYearProbabilityCollection(_probabilities)
            for _year, _probabilities in _prob_dictionary.items()
        }

    def _create_sg_measure(self, measure_result: OrmMeasureResult) -> list[SgMeasure]:
        _mech_year_coll = self._get_mechanism_year_collection(
            measure_result.measure_result_mechanisms,
            SgMeasure.get_allowed_mechanisms(),
        )

        _sg_measures = []
        for _section_result in measure_result.sections_measure_result:
            _cost = _section_result.cost
            _time = _section_result.time
            _lcc = _cost / (1 + self.discount_rate) ** _time

            _sg_measures.append(
                SgMeasure(
                    measure_type=MeasureTypeEnum.get_enum(
                        measure_result.measure_per_section.measure.measure_type.name
                    ),
                    combine_type=CombinableTypeEnum.get_enum(
                        measure_result.measure_per_section.measure.combinable_type.name
                    ),
                    cost=_cost,
                    year=_time,
                    lcc=_lcc,
                    mechanism_year_collection=_mech_year_coll.get(
                        _time, MechanismPerYearProbabilityCollection([])
                    ),
                    dberm=self.get_parameter(measure_result, "dberm"),
                    dcrest=self.get_parameter(measure_result, "dcrest"),
                )
            )

        return _sg_measures

    def _create_sh_measure(self, measure_result: OrmMeasureResult) -> list[ShMeasure]:
        _mech_year_coll = self._get_mechanism_year_collection(
            measure_result.measure_result_mechanisms,
            ShMeasure.get_allowed_mechanisms(),
        )
        _sh_measures = []
        for _section_result in measure_result.sections_measure_result:
            _cost = _section_result.cost
            _time = _section_result.time
            _lcc = _cost / (1 + self.discount_rate) ** _time

            _sh_measures.append(
                ShMeasure(
                    measure_type=MeasureTypeEnum.get_enum(
                        measure_result.measure_per_section.measure.measure_type.name
                    ),
                    combine_type=CombinableTypeEnum.get_enum(
                        measure_result.measure_per_section.measure.combinable_type.name
                    ),
                    cost=_cost,
                    year=_time,
                    lcc=_lcc,
                    mechanism_year_collection=_mech_year_coll.get(
                        _time, MechanismPerYearProbabilityCollection([])
                    ),
                    beta_target=self.get_parameter(measure_result, "beta_target"),
                    transition_level=self.get_parameter(
                        measure_result, "transition_level"
                    ),
                    dcrest=self.get_parameter(measure_result, "dcrest"),
                )
            )
        return _sh_measures

    def import_orm(self, orm_model: OrmMeasureResult) -> list[MeasureAsInputProtocol]:

        if not orm_model:
            raise ValueError(f"No valid value given for {OrmMeasure.__name__}.")

        _imported_measures = []
        if self.valid_parameter(orm_model, "dberm"):
            _imported_measures.extend(self._create_sh_measure(orm_model))

        if self.valid_parameter(orm_model, "dcrest"):
            _imported_measures.extend(self._create_sg_measure(orm_model))

        return _imported_measures
