import math
from copy import deepcopy
from typing import Iterator

from vrtool.common.enums.combinable_type_enum import CombinableTypeEnum
from vrtool.common.enums.measure_type_enum import MeasureTypeEnum
from vrtool.common.enums.mechanism_enum import MechanismEnum
from vrtool.common.measure_unit_costs import MeasureUnitCosts
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
from vrtool.optimization.measures.sh_sg_measure import ShSgMeasure
from vrtool.orm.io.importers.orm_importer_protocol import OrmImporterProtocol
from vrtool.orm.models.measure import Measure as OrmMeasure
from vrtool.orm.models.measure_result.measure_result import (
    MeasureResult as OrmMeasureResult,
)
from vrtool.orm.models.measure_result.measure_result_section import (
    MeasureResultSection as OrmMeasureResultSection,
)
from vrtool.orm.models.mechanism_per_section import MechanismPerSection
from vrtool.probabilistic_tools.probabilistic_functions import beta_to_pf


class OptimizationMeasureResultImporter(OrmImporterProtocol):
    """
    This importer focuses on the creation of Sh / Sg measures for a given
    `MeasureResult`. However, it does not set its real (start) cost, as this
    depends on which other measures have been previously imported.
    """

    discount_rate: float
    unit_costs: MeasureUnitCosts
    investment_years: list[int]

    def __init__(
        self, vrtool_config: VrtoolConfig, investment_years: list[int]
    ) -> None:
        if not vrtool_config:
            raise ValueError("VrtoolConfig not provided.")

        self.discount_rate = vrtool_config.discount_rate
        self.unit_costs = vrtool_config.unit_costs
        self.investment_years = investment_years

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
        measure_as_input_type: type[MeasureAsInputProtocol],
    ) -> list[MeasureAsInputProtocol]:
        _mech_year_coll = self._get_mechanism_year_collection(
            measure_result.measure_result_mechanisms,
            measure_as_input_type.get_allowed_mechanisms(),
        )
        _measure_concrete_params = measure_as_input_type.get_concrete_parameters()
        _measures_dicts = []
        for _section_result in measure_result.measure_result_section.where(
            OrmMeasureResultSection.time == 0
        ):
            # Create a measure for each investment year (VRTOOL-431).
            for _year in self.investment_years:
                _cost = _section_result.cost
                _measures_dicts.append(
                    dict(
                        measure_result_id=measure_result.id,
                        measure_type=MeasureTypeEnum.get_enum(
                            measure_result.measure_per_section.measure.measure_type.name
                        ),
                        combine_type=CombinableTypeEnum.get_enum(
                            measure_result.measure_per_section.measure.combinable_type.name
                        ),
                        cost=_cost,
                        year=_year,
                        discount_rate=self.discount_rate,
                        mechanism_year_collection=deepcopy(_mech_year_coll),
                    )
                    | {
                        _param: measure_result.get_parameter_value(_param)
                        for _param in _measure_concrete_params
                    }
                )

        return list(map(lambda x: measure_as_input_type(**x), _measures_dicts))

    @staticmethod
    def get_measure_as_input_types(
        measure_result: OrmMeasureResult,
    ) -> Iterator[type[MeasureAsInputProtocol]]:
        """
        Gets the corresponding imported type(s) for a `MeasureResult`.
        It could also be that no type is available for the given `MeasureResult`.

        Args:
            measure_result (OrmMeasureResult): Measure result to import.

        Yields:
            Iterator[type[MeasureAsInputProtocol]]: Iterator of types that can be used to import the given measure result.
        """

        def valid_parameter(parameter_name: str) -> bool:
            _parameter_value = measure_result.get_parameter_value(parameter_name)
            return math.isclose(_parameter_value, 0) or math.isnan(_parameter_value)

        _combinable_type = CombinableTypeEnum.get_enum(
            measure_result.combinable_type_name
        )
        if ShMeasure.is_combinable_type_allowed(_combinable_type) and valid_parameter(
            "dberm"
        ):
            yield ShMeasure

        if SgMeasure.is_combinable_type_allowed(_combinable_type) and valid_parameter(
            "dcrest"
        ):
            yield SgMeasure

        if measure_result.measure_type == MeasureTypeEnum.CUSTOM:
            yield ShSgMeasure

    def import_orm(self, orm_model: OrmMeasureResult) -> list[MeasureAsInputProtocol]:

        if not orm_model:
            raise ValueError(f"No valid value given for {OrmMeasure.__name__}.")

        _imported_measures = []

        for _mip_type in self.get_measure_as_input_types(orm_model):
            _imported_measures.extend(self._create_measure(orm_model, _mip_type))

        if not _imported_measures:
            _imported_measures.extend(self._create_measure(orm_model, ShSgMeasure))

        return _imported_measures
