import math
from typing import Iterator

from vrtool.common.enums.combinable_type_enum import CombinableTypeEnum
from vrtool.common.enums.measure_type_enum import MeasureTypeEnum
from vrtool.common.measure_unit_costs import MeasureUnitCosts
from vrtool.defaults.vrtool_config import VrtoolConfig
from vrtool.optimization.measures.measure_as_input_protocol import (
    MeasureAsInputProtocol,
)
from vrtool.optimization.measures.sg_measure import SgMeasure
from vrtool.optimization.measures.sh_measure import ShMeasure
from vrtool.optimization.measures.sh_sg_measure import ShSgMeasure
from vrtool.orm.io.importers.optimization.measures.measure_as_input_collection_importer import (
    MeasureAsInputCollectionImporter,
)
from vrtool.orm.io.importers.optimization.measures.measure_as_input_importer_data import (
    MeasureAsInputImporterData,
)
from vrtool.orm.io.importers.orm_importer_protocol import OrmImporterProtocol
from vrtool.orm.models.measure import Measure as OrmMeasure
from vrtool.orm.models.measure_result.measure_result import (
    MeasureResult as OrmMeasureResult,
)


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
        self,
        vrtool_config: VrtoolConfig,
        investment_years: list[int],
    ) -> None:
        if not vrtool_config:
            raise ValueError("VrtoolConfig not provided.")

        self.discount_rate = vrtool_config.discount_rate
        self.unit_costs = vrtool_config.unit_costs
        self.investment_years = investment_years

    @staticmethod
    def get_measure_as_input_importer_data(
        measure_result: OrmMeasureResult,
    ) -> Iterator[type[MeasureAsInputImporterData]]:
        """
        Gets the corresponding importer type(s) for a `MeasureResult`.
        It could also be that no type is available for the given `MeasureResult`.

        Args:
            measure_result (OrmMeasureResult): Measure result to import.

        Yields:
            Iterator[type[MeasureAsInputBaseImporter]]: Iterator of importer types
              that can be used to import the given measure result.
        """

        def parameter_not_relevant(parameter_name: str) -> bool:
            _parameter_value = measure_result.get_parameter_value(parameter_name)
            return math.isclose(_parameter_value, 0) or math.isnan(_parameter_value)

        _combinable_type = CombinableTypeEnum.get_enum(
            measure_result.combinable_type_name
        )
        if ShMeasure.is_combinable_type_allowed(
            _combinable_type
        ) and parameter_not_relevant("dberm"):
            yield MeasureAsInputImporterData(
                measure_as_input_type=ShMeasure,
                concrete_parameters=[
                    "beta_target",
                    "transition_level",
                    "dcrest",
                    "l_stab_screen",
                ],
            )

        if SgMeasure.is_combinable_type_allowed(
            _combinable_type
        ) and parameter_not_relevant("dcrest"):
            yield MeasureAsInputImporterData(
                measure_as_input_type=SgMeasure,
                concrete_parameters=[
                    "dberm",
                    "l_stab_screen",
                ],
            )

        if measure_result.measure_type == MeasureTypeEnum.CUSTOM:
            # VRTOOL-518: To avoid not knowing which MeasureResult.id needs to be
            # selected we opted to generate a ShSgMeasure to solve this issue.
            # However, this will imply the creation of "too many" Custom
            # `ShSgMeasure` which is accepted for now.
            yield MeasureAsInputImporterData(
                measure_as_input_type=ShSgMeasure, concrete_parameters=[]
            )

    def import_orm(self, orm_model: OrmMeasureResult) -> list[MeasureAsInputProtocol]:

        if not orm_model:
            raise ValueError(f"No valid value given for {OrmMeasure.__name__}.")

        _imported_measures = []

        for _mip_importer_data in self.get_measure_as_input_importer_data(orm_model):
            _imported_measures.extend(
                MeasureAsInputCollectionImporter(
                    _mip_importer_data
                ).import_measure_as_input_collection()
            )

        if not _imported_measures:
            _shsg_importer = MeasureAsInputCollectionImporter(
                MeasureAsInputImporterData(
                    measure_as_input_type=ShSgMeasure,
                    concrete_parameters=[],
                    measure_result=orm_model,
                    investment_years=self.investment_years,
                    discount_rate=self.discount_rate,
                )
            )
            _imported_measures.extend(
                _shsg_importer.import_measure_as_input_collection()
            )

        return _imported_measures
