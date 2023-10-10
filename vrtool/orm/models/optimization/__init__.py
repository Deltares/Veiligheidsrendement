from vrtool.orm.models.optimization.optimization_run import OptimizationRun
from vrtool.orm.models.optimization.optimization_selected_measure import (
    OptimizationSelectedMeasure,
)
from vrtool.orm.models.optimization.optimization_step import OptimizationStep
from vrtool.orm.models.optimization.optimization_step_result_mechanism import (
    OptimizationStepResultMechanism,
)
from vrtool.orm.models.optimization.optimization_step_result_section import (
    OptimizationStepResultSection,
)
from vrtool.orm.models.optimization.optimization_type import OptimizationType


def get_optimization_results_tables() -> list:
    return [
        OptimizationType,
        OptimizationRun,
        OptimizationStep,
        OptimizationSelectedMeasure,
        OptimizationStepResultMechanism,
        OptimizationStepResultSection,
    ]
