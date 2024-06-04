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
from vrtool.orm.io.importers.optimization.measures.measure_as_input_base_importer import (
    MeasureAsInputBaseImporter,
)
from vrtool.orm.io.importers.optimization.measures.sg_measure_importer import (
    SgMeasureImporter,
)
from vrtool.orm.io.importers.optimization.measures.sh_measure_importer import (
    ShMeasureImporter,
)
from vrtool.orm.io.importers.optimization.measures.sh_sg_measure_importer import (
    ShSgMeasureImporter,
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
    def get_measure_as_input_importer_type(
        measure_result: OrmMeasureResult,
    ) -> Iterator[type[MeasureAsInputBaseImporter]]:
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
            yield ShMeasureImporter

        if SgMeasure.is_combinable_type_allowed(
            _combinable_type
        ) and parameter_not_relevant("dcrest"):
            yield SgMeasureImporter

        if measure_result.measure_type == MeasureTypeEnum.CUSTOM:
            # VRTOOL-518: To avoid not knowing which MeasureResult.id needs to be
            # selected we opted to generate a ShSgMeasure to solve this issue.
            # However, this will imply the creation of "too many" Custom
            # `ShSgMeasure` which is accepted for now.
            yield ShSgMeasureImporter

    def import_orm(self, orm_model: OrmMeasureResult) -> list[MeasureAsInputProtocol]:

        if not orm_model:
            raise ValueError(f"No valid value given for {OrmMeasure.__name__}.")

        _imported_measures = []

        for _mip_importer_type in self.get_measure_as_input_importer_type(orm_model):
            _mip_importer = _mip_importer_type(
                orm_model, self.investment_years, self.discount_rate
            )
            _imported_measures.extend(_mip_importer.create_measure())

        if not _imported_measures:
            _shsg_importer = ShSgMeasureImporter(
                orm_model, self.investment_years, self.discount_rate
            )
            _imported_measures.extend(_shsg_importer.create_measure())

        return _imported_measures
