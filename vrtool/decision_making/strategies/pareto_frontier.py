import copy

import numpy as np
import pandas as pd
from vrtool.decision_making.strategies.strategy_base import StrategyBase
from vrtool.decision_making.strategies.mixed_integer_strategy import MixedIntegerStrategy
from tools.HelperFunctions import get_measure_table

class ParetoFrontier(StrategyBase):
    """This is a subclass for generating a ParetoFrontier based on Mixed Integer evaluations with a budget limit."""

    def evaluate(self, traject_obj, solutions_collection, lcc_list: bool=False, strategy_path=False):
        if not lcc_list:
            print()
            # run optimization and generate list on that
        MIPObject = MixedIntegerStrategy("MIPObject")
        MIPObject.combine(traject_obj, solutions_collection, splitparams=True)
        MIPObject.make_optimization_input(traject_obj, solutions_collection)

        MIPObjects = []
        MIPModels = []
        MIPResults = []
        LCC = []
        TR = []
        TC = []
        ObjectiveValue = []
        for j in range(0, len(lcc_list)):
            MIPObjects.append(copy.deepcopy(MIPObject))
            MIPModels.append(
                MIPObjects[-1].create_optimization_model(BudgetLimit=lcc_list[j])
            )
            MIPModels[-1].solve()
            MIPResult = {}
            MIPResult["Values"] = MIPModels[-1].solution.get_values()
            MIPResult["Names"] = MIPModels[-1].variables.get_names()
            MIPResult["ObjectiveValue"] = MIPModels[-1].solution.get_objective_value()
            MIPResult["Status"] = MIPModels[-1].solution.get_status_string()
            MIPResults.append(MIPResult)
            MIPObjects[-1].readResults(
                MIPResults[-1], MeasureTable=get_measure_table(solutions_collection)
            )
            MIPObjects[-1].TakenMeasures.to_csv(
                strategy_path.joinpath("Pareto_LCC=" + str(np.int32(lcc_list[j])) + ".csv")
            )
            LCC.append(MIPObjects[-1].results["LCC"])
            TR.append(
                MIPObjects[-1].results["GeoRisk"]
                + MIPObjects[-1].results["OverflowRisk"]
            )
            TC.append(MIPObjects[-1].results["TC"])
            ObjectiveValue.append(MIPResult["ObjectiveValue"])
            print(MIPModels[-1].solution.status[MIPModels[-1].solution.get_status()])

        self.costs = pd.DataFrame(
            np.array([LCC, TR, TC]).T, columns=["LCC", "TR", "TC"]
        )
        # print(LCC)
        # print(TR)
        #
        # print(ObjectiveValue)
        # Summarize results:
        # print csvs of TakenMeasures with name: path + Pareto_LCC= LCClist[j]
        # Generate TCs_pareto (LCC, TR, TC)

