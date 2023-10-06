from vrtool.decision_making.strategies.strategy_base import StrategyBase
from vrtool.orm.io.exporters.orm_exporter_protocol import OrmExporterProtocol
from vrtool.orm.models.measure_result.measure_result_mechanism import MeasureResultMechanism
from vrtool.orm.models.optimization.optimization_step import OptimizationStep
from vrtool.orm.models.optimization.optimization_step_result import OptimizationStepResult


class StrategyBaseExporter(OrmExporterProtocol):

    def __init__(self) -> None:
        self._opt_step_id = 0

    def export_dom(self, dom_model: StrategyBase) -> None:
        dims = dom_model.TakenMeasures.values.shape
        steps = []
        stepResults = []

        cntMeasuresPerSection = {}
        sumMeasures = 0
        for section in dom_model.indexCombined2single:
            cntMeasuresPerSection[section] = sumMeasures
            singlesMeasures = max(dom_model.indexCombined2single[section])
            sumMeasures += singlesMeasures[0]

        for i in range(1, dims[0]):
            section = dom_model.TakenMeasures.values[i,0]
            measure_id = dom_model.TakenMeasures.values[i,1]
            splittedMeasures = dom_model.indexCombined2single[section][measure_id]
            for singleMsrId in splittedMeasures:
                msrId = singleMsrId+cntMeasuresPerSection[section]
                msr = dom_model.options[section].values[singleMsrId]
                lcc = dom_model.TakenMeasures.values[i,2]
                steps.append({"step_number":i, "optimization_selected_measure_id":msrId})
                self._opt_step_id += 1
                offset = len(msr) - len(dom_model.T)
                for j in range(len(dom_model.T)):
                    t  = dom_model.T[j]
                    beta = msr[offset+j]
                    mechanism_per_section_id = -1 # TODO value for combined mechanisms
                    stepResults.append({"optimization_step_id":self._opt_step_id,
                                        "mechanism_per_section_id": mechanism_per_section_id,
                                        "time": t, "beta": beta, "lcc":lcc })

                _measure_result = MeasureResult.get_by_id(msrId)
                rows = _measure_result.measure_result_mechanisms
                for row in rows:
                    print(row.mechanism_per_section_id, row.beta, row.time)
                    stepResults.append({"optimization_step_id":self._opt_step_id,
                                        "mechanism_per_section_id": row.mechanism_per_section_id,
                                        "time": row.time, "beta": row.beta, "lcc":lcc })

        OptimizationStep.insert_many(steps).execute()

        OptimizationStepResult.insert_many(stepResults).execute()
