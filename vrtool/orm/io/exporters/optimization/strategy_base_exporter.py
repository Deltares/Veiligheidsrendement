from vrtool.decision_making.strategies.strategy_base import StrategyBase
from vrtool.orm.io.exporters.orm_exporter_protocol import OrmExporterProtocol
from vrtool.orm.models.measure_result.measure_result import MeasureResult
from vrtool.orm.models.optimization import (
    OptimizationStep,
    OptimizationStepResultMechanism,
    OptimizationStepResultSection,
)


class StrategyBaseExporter(OrmExporterProtocol):
    def __init__(self) -> None:
        self._opt_step_id = 0

    def export_dom(self, dom_model: StrategyBase) -> None:
        dims = dom_model.TakenMeasures.values.shape
        _step_results_section = []
        _step_results_mechanism = []

        cntMeasuresPerSection = {}
        sumMeasures = 0
        for section in dom_model.indexCombined2single:
            cntMeasuresPerSection[section] = sumMeasures
            singlesMeasures = max(dom_model.indexCombined2single[section])
            sumMeasures += singlesMeasures[0]

        for i in range(1, dims[0]):
            section = dom_model.TakenMeasures.values[i, 0]
            measure_id = dom_model.TakenMeasures.values[i, 1]
            splittedMeasures = dom_model.indexCombined2single[section][measure_id]
            for singleMsrId in splittedMeasures:
                msrId = singleMsrId + cntMeasuresPerSection[section]
                msr = dom_model.options[section].values[singleMsrId]
                lcc = dom_model.TakenMeasures.values[i, 2]
                offset = len(msr) - len(dom_model.T)
                _created_optimization_step = OptimizationStep.create(
                    step_number=i,
                    optimization_selected_measure_id=msrId,
                )
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

                _measure_result = MeasureResult.get_by_id(msrId)
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
