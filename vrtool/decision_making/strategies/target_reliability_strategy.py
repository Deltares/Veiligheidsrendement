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
        n_piping = 1 + (
            traject.general_info.aPiping
            * traject.general_info.TrajectLength
            / traject.general_info.bPiping
        )
        n_stab = 1 + (
            traject.general_info.aStabilityInner
            * traject.general_info.TrajectLength
            / traject.general_info.bStabilityInner
        )
        n_overflow = 1
        beta_cs_piping = pf_to_beta(
            traject.general_info.Pmax * traject.general_info.omegaPiping / n_piping
        )
        n_revetment = 3
        omegaRevetment = 0.1
        beta_cs_revetment = pf_to_beta(
            traject.general_info.Pmax * omegaRevetment / n_revetment
        )
        beta_cs_stabinner = pf_to_beta(
            traject.general_info.Pmax
            * traject.general_info.omegaStabilityInner
            / n_stab
        )
        beta_cs_overflow = pf_to_beta(
            traject.general_info.Pmax * traject.general_info.omegaOverflow / n_overflow
        )

        # Rank sections based on 2075 Section probability
        beta_horizon = []
        for i in traject.sections:
            # For now (VRTOOL-221) `OI_horizon` is assumed to be a string.
            _oi_horizon = (
                self.OI_horizon
                if isinstance(self.OI_horizon, str)
                else str(self.OI_horizon)
            )
            beta_horizon.append(
                i.section_reliability.SectionReliability.loc["Section"][_oi_horizon]
            )

        section_indices = np.argsort(beta_horizon)
        measure_cols = ["Section", "option_index", "LCC", "BC"]

        if splitparams:
            _taken_measures = pd.DataFrame(
                data=[[None, None, 0, None, None, None, None, None, None, None, None]],
                columns=measure_cols
                + [
                    "ID",
                    "name",
                    "yes/no",
                    "dcrest",
                    "dberm",
                    "beta_target",
                    "transition_level",
                ],
            )
        else:
            _taken_measures = pd.DataFrame(
                data=[[None, None, None, 0, None, None, None]],
                columns=measure_cols + ["ID", "name", "params"],
            )
        # columns (section name and index in self.options[section])
        _base_traject_probability = make_traject_df(traject, cols)
        _probability_steps = [copy.deepcopy(_base_traject_probability)]
        _traject_probability = copy.deepcopy(_base_traject_probability)

        for j in section_indices:
            i = traject.sections[j]
            # convert beta_cs to beta_section in order to correctly search self.options[section]
            # TODO THIS IS CURRENTLY INCONSISTENT WITH THE WAY IT IS CALCULATED: it should be coupled to whether the length effect within sections is turned on or not
            if self.LE_in_section:
                logging.warning(
                    "In evaluate for TargetReliabilityStrategy: THIS CODE ON LENGTH EFFECT WITHIN SECTIONS SHOULD BE TESTED"
                )
                _beta_t_piping = pf_to_beta(
                    beta_to_pf(beta_cs_piping)
                    * (i.Length / traject.general_info.bPiping)
                )
                _beta_t_sabinner = pf_to_beta(
                    beta_to_pf(beta_cs_stabinner)
                    * (i.Length / traject.general_info.bStabilityInner)
                )
            else:
                _beta_t_piping = beta_cs_piping
                _beta_t_sabinner = beta_cs_stabinner
            _beta_t_overflow = beta_cs_overflow
            _beta_t_revetment = beta_cs_revetment
            _beta_t = {
                "Piping": _beta_t_piping,
                "StabilityInner": _beta_t_sabinner,
                "Overflow": _beta_t_overflow,
                "Revetment": _beta_t_revetment,
            }
            # find cheapest design that satisfies betatcs in 50 years from OI_year if OI_year is an int that is not 0
            if isinstance(self.OI_year, int):
                # TODO: should this not be OI_year + 50?
                _target_year = 50

            # make PossibleMeasures dataframe
            _possible_measures = copy.deepcopy(self.options[i.name])
            # filter for mechanisms that are considered
            for mechanism in traject.mechanism_names:
                _possible_measures = _possible_measures.loc[
                    self.options[i.name][(mechanism, _target_year)] > _beta_t[mechanism]
                ]

            if len(_possible_measures) == 0:
                # continue to next section if weakest has no more measures
                logging.warning(
                    "Warning: for Target reliability strategy no suitable measures were found for section {}".format(
                        i.name
                    )
                )
                continue
            # calculate LCC
            _lcc = calc_tc(
                _possible_measures,
                self.discount_rate,
                horizon=self.options[i.name]["Overflow"].columns[-1],
            )

            # select measure with lowest cost
            idx = np.argmin(_lcc)

            measure = _possible_measures.iloc[idx]
            option_index = _possible_measures.index[idx]
            # calculate achieved risk reduction & BC ratio compared to base situation
            _r_base, _dr, _t_r = calc_tr(
                i.name,
                measure,
                _traject_probability,
                original_section=_traject_probability.loc[i.name],
                discount_rate=self.discount_rate,
                horizon=cols[-1],
                damage=traject.general_info.FloodDamage,
            )
            _bc = _dr / _lcc[idx]

            if splitparams:
                name = id_to_name(
                    measure["ID"].values[0], solutions_dict[i.name].measure_table
                )
                data_opt = pd.DataFrame(
                    [
                        [
                            i.name,
                            option_index,
                            _lcc[idx],
                            _bc,
                            measure["ID"].values[0],
                            name,
                            measure["yes/no"].values[0],
                            measure["dcrest"].values[0],
                            measure["dberm"].values[0],
                            measure["beta_target"].values[0],
                            measure["transition_level"].values[0],
                        ]
                    ],
                    columns=_taken_measures.columns,
                )
            else:
                data_opt = pd.DataFrame(
                    [
                        [
                            i.name,
                            option_index,
                            _lcc[idx],
                            _bc,
                            measure["ID"].values[0],
                            measure["name"].values[0],
                            measure["params"].values[0],
                        ]
                    ],
                    columns=_taken_measures.columns,
                )  # here we evaluate and pick the option that has the
                # lowest total cost and a BC ratio that is lower than any measure at any other section

            # Add to TakenMeasures
            _taken_measures = pd.concat((_taken_measures, data_opt))
            # Calculate new probabilities
            _traject_probability = implement_option(
                i.name, _traject_probability, measure
            )
            _probability_steps.append(copy.deepcopy(_traject_probability))
        self.TakenMeasures = _taken_measures
        self.Probabilities = _probability_steps
