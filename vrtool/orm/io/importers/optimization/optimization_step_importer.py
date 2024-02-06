from collections import defaultdict
from typing import Any

import pandas as pd

from vrtool.common.enums.mechanism_enum import MechanismEnum
from vrtool.orm.io.importers.orm_importer_protocol import OrmImporterProtocol
from vrtool.orm.models.optimization.optimization_step import OptimizationStep
from vrtool.orm.models.optimization.optimization_step_result_mechanism import (
    OptimizationStepResultMechanism,
)
from vrtool.orm.models.optimization.optimization_step_result_section import (
    OptimizationStepResultSection,
)


class OptimizationStepImporter(OrmImporterProtocol):
    @staticmethod
    def import_optimization_step_results_df(
        optimization_step: OptimizationStep,
    ) -> pd.DataFrame:
        """
        Imports the section and mechanism results of an `OptimizationStep` in the
        shape of a `pandas.DataFrame`.
        It does not include the `lcc` value.
        """
        _columns = []
        _step_result_dict = defaultdict(list)
        for _osrs in optimization_step.optimization_step_results_section.order_by(
            OptimizationStepResultSection.time.asc()
        ):
            _columns.append(str(_osrs.time))
            _step_result_dict["Section"].append(_osrs.beta)
            for _osrm in optimization_step.optimization_step_results_mechanism.where(
                OptimizationStepResultMechanism.time == _osrs.time
            ):
                _mech_name = MechanismEnum.get_enum(
                    _osrm.mechanism_per_section.mechanism.name
                ).name
                _step_result_dict[_mech_name].append(_osrm.beta)

        return pd.DataFrame.from_dict(
            _step_result_dict, columns=_columns, orient="index"
        )

    def import_orm(self, optimization_step: OptimizationStep) -> Any:
        raise NotImplementedError()
