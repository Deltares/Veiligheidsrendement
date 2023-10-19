from vrtool.decision_making.strategies.strategy_base import StrategyBase
from vrtool.orm.io.exporters.orm_exporter_protocol import OrmExporterProtocol
from vrtool.orm.models.measure_result.measure_result import MeasureResult
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
                msr = dom_model.options[section].values[localId]
                lcc = dom_model.TakenMeasures.values[i, 2]
                offset = len(msr) - len(dom_model.T)
                for j in range(len(dom_model.T)):
                    t = dom_model.T[j]
                    beta = msr[offset + j]
                    _step_results_section.append(
                        {
                            "optimization_step": _created_optimization_step,
                            "time": t,
                            "beta": beta,
                            "lcc": lcc,
                        }
                    )

                _measure_result = MeasureResult.get_by_id(singleMsrId)
                rows = _measure_result.measure_result_mechanisms
                for row in rows:
                    _step_results_mechanism.append(
                        {
                            "optimization_step": _created_optimization_step,
                            "mechanism_per_section_id": row.mechanism_per_section_id,
                            "time": row.time,
                            "beta": row.beta,
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
            if run_measure_result.measure_result_id == single_msr_id:
                return run_measure_result.get_id()

        run_id = self.optimization_run.get_id()
        raise ValueError(
            "OptimizationSelectedMeasure with run_id {} and measure result id {} not found".format(
                run_id, single_msr_id
            )
        )
