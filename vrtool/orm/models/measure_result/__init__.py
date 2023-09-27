from vrtool.orm.models.measure_result.measure_result import MeasureResult
from vrtool.orm.models.measure_result.measure_result_parameter import (
    MeasureResultParameter,
)
from vrtool.orm.models.measure_result.measure_result_mechanism import MeasureResultMechanism
from vrtool.orm.models.measure_result.measure_result_section import MeasureResultSection

def get_orm_tables() -> list:
    return [MeasureResult, MeasureResultParameter, MeasureResultMechanism, MeasureResultSection]