import logging
from vrtool.common.enums import MechanismEnum
from vrtool.decision_making.strategies.strategy_base import StrategyBase
from vrtool.orm.io.exporters.orm_exporter_protocol import OrmExporterProtocol
from vrtool.orm.models.measure_result.measure_result import MeasureResult
from vrtool.orm.models.mechanism import Mechanism
from vrtool.orm.models.mechanism_per_section import MechanismPerSection
from vrtool.orm.models.optimization import (
    OptimizationStep,
    OptimizationStepResultMechanism,
    OptimizationStepResultSection,
)
from vrtool.orm.models.optimization.optimization_run import OptimizationRun


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
                logging.warning("Found measure for section without measures; section: " + section)
                continue
            measure_id = dom_model.TakenMeasures.values[i, 1]
            splittedMeasures = dom_model.indexCombined2single[section][measure_id]
            _total_lcc, _total_risk = dom_model.get_total_lcc_and_risk(i)
            for singleMsrId in splittedMeasures:

                opt_sel_msr_id = self._get_sel_msr_id(singleMsrId)
                _created_optimization_step = OptimizationStep.create(
                    step_number=i,
                    optimization_selected_measure_id=opt_sel_msr_id,
                    total_lcc=_total_lcc,
                    total_risk=_total_risk,
                )

                localId = self._find_id_in_section(
                    singleMsrId, dom_model.indexCombined2single[section]
                )
                beta_section = dom_model.options[section]["Section"].values[localId]
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
                _measure_result = MeasureResult.get_by_id(OptimizationSelectedMeasure.get_by_id(opt_sel_msr_id).measure_result_id)
                rows = _measure_result.measure_result_mechanisms
                for row in rows:
                    mechanism_per_section = MechanismPerSection.get_by_id(row.mechanism_per_section_id)
                    mechanismName = MechanismEnum.get_enum(mechanism_per_section.mechanism.name).name
                    beta_mechanism = dom_model.options[section][mechanismName].values[localId]
                    beta = self._get_selected_time(row.time, beta_mechanism)
                    _step_results_mechanism.append(
                        {
                            "optimization_step": _created_optimization_step,
                            "mechanism_per_section_id": row.mechanism_per_section_id,
                            "time": row.time,
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

    def _get_sel_msr_id(self, single_msr_id) -> int:
        for (
            run_measure_result
        ) in self.optimization_run.optimization_run_measure_results:
            if run_measure_result.id == single_msr_id:
                return run_measure_result.get_id()

        run_id = self.optimization_run.get_id()
        raise ValueError(
            "OptimizationSelectedMeasure with run_id {} and measure result id {} not found".format(
                run_id, single_msr_id
            )
        )
    
    def _get_selected_time(self, t:int, values:list[float]) -> float:
        # fix for t=100 where 99 is the last
        if (t < values.shape[0]-1):
            return values[t]
        return values[-1]

