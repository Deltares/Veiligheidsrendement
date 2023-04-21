import copy
import logging
from typing import Dict

import numpy as np
import pandas as pd

from vrtool.decision_making.solutions import Solutions
from vrtool.decision_making.strategies.strategy_base import StrategyBase
from vrtool.decision_making.strategy_evaluation import (
    calc_tc,
    calc_tr,
    implement_option,
    make_traject_df,
)
from vrtool.flood_defence_system.dike_traject import DikeTraject
from vrtool.probabilistic_tools.probabilistic_functions import beta_to_pf, pf_to_beta


class TargetReliabilityStrategy(StrategyBase):
    """Subclass for evaluation in accordance with basic OI2014 approach.
    This ensures that for a certain time horizon, each section satisfies the cross-sectional target reliability"""

    def evaluate(
        self,
        traject: DikeTraject,
        solutions_dict: Dict[str, Solutions],
        splitparams=False,
    ):
        def id_to_name(found_id, measure_table):
            """
            Previously in tools. Only used once within this evaluate method.
            """
            return measure_table.loc[measure_table["ID"] == found_id]["Name"].values[0]

        cols = list(
            solutions_dict[list(solutions_dict.keys())[0]]
            .MeasureData["Section"]
            .columns.values
        )
        # compute cross sectional requirements
        N_piping = 1 + (
            traject.general_info.aPiping
            * traject.general_info.TrajectLength
            / traject.general_info.bPiping
        )
        N_stab = 1 + (
            traject.general_info.aStabilityInner
            * traject.general_info.TrajectLength
            / traject.general_info.bStabilityInner
        )
        N_overflow = 1
        beta_cs_piping = pf_to_beta(
            traject.general_info.Pmax * traject.general_info.omegaPiping / N_piping
        )
        beta_cs_stabinner = pf_to_beta(
            traject.general_info.Pmax
            * traject.general_info.omegaStabilityInner
            / N_stab
        )
        beta_cs_overflow = pf_to_beta(
            traject.general_info.Pmax * traject.general_info.omegaOverflow / N_overflow
        )

        # Rank sections based on 2075 Section probability
        beta_horizon = []
        for i in traject.sections:
            beta_horizon.append(
                i.section_reliability.SectionReliability.loc["Section"][
                    str(self.OI_horizon)
                ]
            )

        section_indices = np.argsort(beta_horizon)
        measure_cols = ["Section", "option_index", "LCC", "BC"]

        if splitparams:
            TakenMeasures = pd.DataFrame(
                data=[[None, None, 0, None, None, None, None, None, None]],
                columns=measure_cols + ["ID", "name", "yes/no", "dcrest", "dberm"],
            )
        else:
            TakenMeasures = pd.DataFrame(
                data=[[None, None, None, 0, None, None, None]],
                columns=measure_cols + ["ID", "name", "params"],
            )
        # columns (section name and index in self.options[section])
        BaseTrajectProbability = make_traject_df(traject, cols)
        Probability_steps = [copy.deepcopy(BaseTrajectProbability)]
        TrajectProbability = copy.deepcopy(BaseTrajectProbability)

        for j in section_indices:
            i = traject.sections[j]
            # convert beta_cs to beta_section in order to correctly search self.options[section]
            # TODO THIS IS CURRENTLY INCONSISTENT WITH THE WAY IT IS CALCULATED: it should be coupled to whether the length effect within sections is turned on or not
            if self.LE_in_section:
                logging.warn(
                    "In evaluate for TargetReliabilityStrategy: THIS CODE ON LENGTH EFFECT WITHIN SECTIONS SHOULD BE TESTED"
                )
                beta_T_piping = pf_to_beta(
                    beta_to_pf(beta_cs_piping)
                    * (i.Length / traject.general_info.bPiping)
                )
                beta_T_stabinner = pf_to_beta(
                    beta_to_pf(beta_cs_stabinner)
                    * (i.Length / traject.general_info.bStabilityInner)
                )
            else:
                beta_T_piping = beta_cs_piping
                beta_T_stabinner = beta_cs_stabinner
            beta_T_overflow = beta_cs_overflow

            # find cheapest design that satisfies betatcs in 50 years from OI_year if OI_year is an int that is not 0
            if isinstance(self.OI_year, int):
                targetyear = 50  # OI_year + 50
            else:
                targetyear = 50

            # filter for overflow
            PossibleMeasures = copy.deepcopy(
                self.options[i.name].loc[
                    self.options[i.name][("Overflow", targetyear)] > beta_T_overflow
                ]
            )

            # filter for piping
            PossibleMeasures = PossibleMeasures.loc[
                self.options[i.name][("Piping", targetyear)] > beta_T_piping
            ]

            # filter for stabilityinner
            PossibleMeasures = PossibleMeasures.loc[
                PossibleMeasures[("StabilityInner", targetyear)] > beta_T_stabinner
            ]
            if len(PossibleMeasures) == 0:
                # continue to next section if weakest has no more measures
                logging.warn(
                    "Warning: for Target reliability strategy no suitable measures were found for section {}".format(
                        i.name
                    )
                )
                continue
            # calculate LCC
            LCC = calc_tc(
                PossibleMeasures,
                self.discount_rate,
                horizon=self.options[i.name]["Overflow"].columns[-1],
            )

            # select measure with lowest cost
            idx = np.argmin(LCC)

            measure = PossibleMeasures.iloc[idx]

            # calculate achieved risk reduction & BC ratio compared to base situation
            R_base, dR, TR = calc_tr(
                i.name,
                measure,
                TrajectProbability,
                original_section=TrajectProbability.loc[i.name],
                discount_rate=self.discount_rate,
                horizon=cols[-1],
                damage=traject.general_info.FloodDamage,
            )
            BC = dR / LCC[idx]

            if splitparams:
                name = id_to_name(
                    measure["ID"].values[0], solutions_dict[i.name].measure_table
                )
                data_opt = pd.DataFrame(
                    [
                        [
                            i.name,
                            idx,
                            LCC[idx],
                            BC,
                            measure["ID"].values[0],
                            name,
                            measure["yes/no"].values[0],
                            measure["dcrest"].values[0],
                            measure["dberm"].values[0],
                        ]
                    ],
                    columns=TakenMeasures.columns,
                )
            else:
                data_opt = pd.DataFrame(
                    [
                        [
                            i.name,
                            idx,
                            LCC[idx],
                            BC,
                            measure["ID"].values[0],
                            measure["name"].values[0],
                            measure["params"].values[0],
                        ]
                    ],
                    columns=TakenMeasures.columns,
                )  # here we evaluate and pick the option that has the
                # lowest total cost and a BC ratio that is lower than any measure at any other section

            # Add to TakenMeasures
            TakenMeasures = pd.concat((TakenMeasures, data_opt))
            # Calculate new probabilities
            TrajectProbability = implement_option(i.name, TrajectProbability, measure)
            Probability_steps.append(copy.deepcopy(TrajectProbability))
        self.TakenMeasures = TakenMeasures
        self.Probabilities = Probability_steps
