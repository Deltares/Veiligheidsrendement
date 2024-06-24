from vrtool.common.measure_unit_costs import MeasureUnitCosts
from vrtool.defaults.vrtool_config import VrtoolConfig
from vrtool.optimization.measures.measure_as_input_protocol import (
    MeasureAsInputProtocol,
)
from vrtool.orm.io.importers.optimization.measures.measure_as_input_importer import (
    MeasureAsInputImporter,
    MeasureAsInputImporterData,
)
from vrtool.orm.io.importers.orm_importer_protocol import OrmImporterProtocol
from vrtool.orm.models.measure import Measure as OrmMeasure
from vrtool.orm.models.measure_result.measure_result import (
    MeasureResult as OrmMeasureResult,
)


class OptimizationMeasureResultImporter(OrmImporterProtocol):
    """
    This importer focuses on the creation of Sh, Sg and ShSg measures for a given
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

    def import_orm(self, orm_model: OrmMeasureResult) -> list[MeasureAsInputProtocol]:

        if not orm_model:
            raise ValueError(f"No valid value given for {OrmMeasure.__name__}.")

        _imported_measures = []
        for (
            _mip_importer_data
        ) in MeasureAsInputImporterData.get_supported_importer_data(
            orm_model, self.investment_years, self.discount_rate
        ):
            _imported_measures.extend(
                MeasureAsInputImporter(
                    _mip_importer_data
                ).import_measure_as_input_collection()
            )

        return _imported_measures
