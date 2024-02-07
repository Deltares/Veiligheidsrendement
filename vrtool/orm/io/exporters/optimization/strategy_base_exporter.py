import logging

from vrtool.common.enums.mechanism_enum import MechanismEnum
from vrtool.decision_making.strategies.strategy_base import StrategyBase
from vrtool.orm.io.exporters.orm_exporter_protocol import OrmExporterProtocol
from vrtool.orm.models.optimization import (
    OptimizationStep,
    OptimizationStepResultMechanism,
    OptimizationStepResultSection,
)
from vrtool.orm.models.optimization.optimization_run import OptimizationRun
from vrtool.orm.models.optimization.optimization_selected_measure import (
    OptimizationSelectedMeasure,
)


class StrategyBaseExporter(OrmExporterProtocol):
    def __init__(self, optimization_run_id: int) -> None:
        self.optimization_run: OptimizationRun = OptimizationRun.get_by_id(
            optimization_run_id
        )

    def export_dom(self, dom_model: StrategyBase) -> None:
        dims = dom_model.TakenMeasures.values.shape
        _step_results_section = []
        _step_results_mechanism = []

        for i in range(1, dims[0]):
            section = dom_model.TakenMeasures.values[i, 0]
            if not any(dom_model.indexCombined2single[section]):
                logging.warning(
                    "Found measure for section without measures; section: " + section
                )
                continue
            measure_id = dom_model.TakenMeasures.values[i, 1]
            split_measures = dom_model.indexCombined2single[section][measure_id]
            _total_lcc, _total_risk = dom_model.get_total_lcc_and_risk(i)
            for single_measure_result_id in split_measures:

                _option_selected_measure_result = (
                    self._get_optimization_selected_measure(single_measure_result_id)
                )
                _created_optimization_step = OptimizationStep.create(
                    step_number=i,
                    optimization_selected_measure=_option_selected_measure_result,
                    total_lcc=_total_lcc,
                    total_risk=_total_risk,
                )

                _local_id = self._find_id_in_section(
                    single_measure_result_id, dom_model.indexCombined2single[section]
                )
                beta_section = dom_model.options[section]["Section"].values[_local_id]
                lcc = dom_model.TakenMeasures.values[i, 2]
                for j in range(len(dom_model.T)):
                    t = dom_model.T[j]
                    beta = self._get_selected_time(t, beta_section)
                    _step_results_section.append(
                        {
                            "optimization_step": _created_optimization_step,
                            "time": t,
                            "beta": beta,
                            "lcc": lcc,
                        }
                    )
                # From OptimizationSelectedMeasure get measure_result_id based on singleMsrId
                for (
                    _measure_result_mechanism
                ) in (
                    _option_selected_measure_result.measure_result.measure_result_mechanisms
                ):
                    _mechanism_name = MechanismEnum.get_enum(
                        _measure_result_mechanism.mechanism_per_section.mechanism.name
                    ).name
                    beta_mechanism = dom_model.options[section][_mechanism_name].values[
                        _local_id
                    ]
                    beta = self._get_selected_time(
                        _measure_result_mechanism.time, beta_mechanism
                    )
                    _step_results_mechanism.append(
                        {
                            "optimization_step": _created_optimization_step,
                            "mechanism_per_section_id": _measure_result_mechanism.mechanism_per_section_id,
                            "time": _measure_result_mechanism.time,
                            "beta": beta,
                            "lcc": lcc,
                        }
                    )

        OptimizationStepResultSection.insert_many(_step_results_section).execute()
        OptimizationStepResultMechanism.insert_many(_step_results_mechanism).execute()

    def _find_id_in_section(self, measure_id: int, index_section: list[int]) -> int:
        for i in range(len(index_section)):
            if index_section[i][0] == measure_id:
                return i
        raise ValueError(
            "Measure ID {} not found in any of the section indices.".format(measure_id)
        )

    def _get_optimization_selected_measure(
        self, single_msr_id: int
    ) -> OptimizationSelectedMeasure:
        _opt_selected_measure = (
            self.optimization_run.optimization_run_measure_results.where(
                OptimizationSelectedMeasure.id == single_msr_id
            ).get_or_none()
        )
        if not _opt_selected_measure:
            raise ValueError(
                "OptimizationSelectedMeasure with run_id {} and measure result id {} not found".format(
                    self.optimization_run.get_id(), single_msr_id
                )
            )
        return _opt_selected_measure

    def _get_selected_time(self, t: int, values: list[float]) -> float:
        # fix for t=100 where 99 is the last
        if t < values.shape[0] - 1:
            return values[t]
        return values[-1]
