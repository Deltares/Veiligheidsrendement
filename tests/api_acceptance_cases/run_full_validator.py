from __future__ import annotations

import vrtool.orm.models as orm
from tests.api_acceptance_cases.run_step_optimization_validator import (
    RunStepOptimizationValidator,
)
from tests.api_acceptance_cases.run_step_validator_protocol import RunStepValidator
from vrtool.defaults.vrtool_config import VrtoolConfig

OptimizationStepResult = (
    orm.OptimizationStepResultMechanism | orm.OptimizationStepResultSection
)

vrtool_db_default_name = "vrtool_input.db"


class RunFullValidator(RunStepValidator):
    def validate_preconditions(self, valid_vrtool_config: VrtoolConfig):
        assert RunStepOptimizationValidator.get_csv_reference_dir(
            valid_vrtool_config
        ).exists()

    def validate_results(self, valid_vrtool_config: VrtoolConfig):
        # Validate the optimization results.
        # TODO: Remove this validator class if we understand that
        # the optimization validation is enough.
        _optimization_validator = RunStepOptimizationValidator()
        _optimization_validator.validate_results(valid_vrtool_config)
