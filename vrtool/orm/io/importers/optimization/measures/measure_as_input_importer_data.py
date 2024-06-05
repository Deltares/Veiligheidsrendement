import math
from dataclasses import dataclass, field

from vrtool.common.enums.measure_type_enum import MeasureTypeEnum
from vrtool.optimization.measures.measure_as_input_protocol import (
    MeasureAsInputProtocol,
)
from vrtool.optimization.measures.sg_measure import SgMeasure
from vrtool.optimization.measures.sh_measure import ShMeasure
from vrtool.optimization.measures.sh_sg_measure import ShSgMeasure
from vrtool.orm.models.measure_result.measure_result import MeasureResult


@dataclass(kw_only=True)
class MeasureAsInputImporterData:
    measure_as_input_type: type[MeasureAsInputProtocol]
    concrete_parameters: list[str]
    measure_result: MeasureResult
    investment_years: list[int] = field(default_factory=list)
    discount_rate: float = float("nan")

    @classmethod
    def get_supported_importer_data(
        cls,
        measure_result: MeasureResult,
        investment_years: list[int],
        discount_rate: float,
    ) -> list["MeasureAsInputImporterData"]:
        """
        Gets all instances of a `MeasureAsInputImporterData` that
        support the given `measure_result` argument (`MeasureResult`).
        """

        def parameter_not_relevant(parameter_name: str) -> bool:
            _parameter_value = measure_result.get_parameter_value(parameter_name)
            return math.isclose(_parameter_value, 0) or math.isnan(_parameter_value)

        _supported_importer_data = []
        if ShMeasure.is_combinable_type_allowed(
            measure_result.combinable_type
        ) and parameter_not_relevant("dberm"):
            _supported_importer_data.append(
                MeasureAsInputImporterData(
                    measure_result=measure_result,
                    measure_as_input_type=ShMeasure,
                    concrete_parameters=[
                        "beta_target",
                        "transition_level",
                        "dcrest",
                        "l_stab_screen",
                    ],
                    investment_years=investment_years,
                    discount_rate=discount_rate,
                )
            )

        if SgMeasure.is_combinable_type_allowed(
            measure_result.combinable_type
        ) and parameter_not_relevant("dcrest"):
            _supported_importer_data.append(
                MeasureAsInputImporterData(
                    measure_result=measure_result,
                    measure_as_input_type=SgMeasure,
                    concrete_parameters=[
                        "dberm",
                        "l_stab_screen",
                    ],
                    investment_years=investment_years,
                    discount_rate=discount_rate,
                )
            )

        if (
            not any(_supported_importer_data)
            or measure_result.measure_type == MeasureTypeEnum.CUSTOM
        ):
            # VRTOOL-518: To avoid not knowing which MeasureResult.id needs to be
            # selected we opted to generate a ShSgMeasure to solve this issue.
            # However, this will imply the creation of "too many" Custom
            # `ShSgMeasure` which is accepted for now.
            _supported_importer_data.append(
                MeasureAsInputImporterData(
                    measure_result=measure_result,
                    measure_as_input_type=ShSgMeasure,
                    concrete_parameters=["dberm", "dcrest", "l_stab_screen"],
                    investment_years=investment_years,
                    discount_rate=discount_rate,
                )
            )
        return _supported_importer_data
