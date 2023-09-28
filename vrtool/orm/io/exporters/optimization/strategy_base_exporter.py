from vrtool.decision_making.strategies.strategy_base import StrategyBase
from vrtool.orm.io.exporters.orm_exporter_protocol import OrmExporterProtocol
from vrtool.orm.models.optimization.optimization_step import OptimizationStep
from vrtool.orm.models.optimization.optimization_step_result import OptimizationStepResult


class StrategyBaseExporter(OrmExporterProtocol):

    def __init__(self) -> None:
        self._opt_step_id = 0

    def export_dom(self, dom_model: StrategyBase) -> None:
        dims = dom_model.TakenMeasures.values.shape
        steps = []
        stepResults = []
        for i in range(1, dims[0]):
            section = dom_model.TakenMeasures.values[i,0]
            measure_id = dom_model.TakenMeasures.values[i,1] # TODO: split combined measures
            msr = dom_model.options[section].values[measure_id]
            lcc = dom_model.TakenMeasures.values[i,2]
            steps.append({"step_number":i, "optimization_selected_measure_id":measure_id})
            offset = len(msr) - len(dom_model.T)
            for j in range(len(dom_model.T)):
                t  = dom_model.T[j]
                beta = msr[offset+j]
                # TODO mechanism_per_section_id not evaluated
                stepResults.append({"optimization_step_id":self._opt_step_id, "mechanism_per_section_id": -999, "time": t, "beta": beta, "lcc":lcc })
            self._opt_step_id += 1

        OptimizationStep.insert_many(steps).execute()

        OptimizationStepResult.insert_many(stepResults).execute()
