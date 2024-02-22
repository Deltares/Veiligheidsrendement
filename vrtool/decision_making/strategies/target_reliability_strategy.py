import copy
import logging
from typing import Dict
from dataclasses import dataclass
import numpy as np
import pandas as pd

from vrtool.common.enums.mechanism_enum import MechanismEnum
from vrtool.decision_making.solutions import Solutions
from vrtool.decision_making.strategies.strategy_protocol import StrategyProtocol
from vrtool.decision_making.strategy_evaluation import (
    calc_tc,
    calc_tr,
    implement_option,
    make_traject_df,
)
from vrtool.defaults.vrtool_config import VrtoolConfig
from vrtool.flood_defence_system.dike_traject import DikeTraject
from vrtool.optimization.strategy_input.strategy_input_target_reliability import (
    StrategyInputTargetReliability,
)
from vrtool.probabilistic_tools.probabilistic_functions import beta_to_pf, pf_to_beta


@dataclass
class CrossSectionalRequirements:
    beta_cs_piping: np.ndarray
    beta_cs_revetment: np.ndarray
    beta_cs_stabinner: np.ndarray
    beta_cs_overflow: np.ndarray

    dike_traject_b_piping: float
    dike_traject_b_stability_inner: float

    @property
    def pf_cs_piping(self) -> np.ndarray:
        return beta_to_pf(self.beta_cs_piping)

    @property
    def pf_cs_stabinner(self) -> np.ndarray:
        return beta_to_pf(self.beta_cs_stabinner)

    def calculate_beta_t_piping(
        self, dike_section_length: float, le_in_section: bool
    ) -> np.ndarray:
        if not le_in_section:
            return self.beta_cs_piping
        return pf_to_beta(
            self.pf_cs_piping * (dike_section_length / self.dike_traject_b_piping)
        )

    def calculate_beta_t_sabinner(
        self, dike_section_length: float, le_in_section: bool
    ) -> np.ndarray:
        if not le_in_section:
            return self.beta_cs_stabinner
        return pf_to_beta(
            self.pf_cs_stabinner
            * (dike_section_length / self.dike_traject_b_stability_inner)
        )

    @classmethod
    def from_dike_traject(cls, dike_traject: DikeTraject):
        # compute cross sectional requirements
        n_piping = 1 + (
            dike_traject.general_info.aPiping
            * dike_traject.general_info.TrajectLength
            / dike_traject.general_info.bPiping
        )
        n_stab = 1 + (
            dike_traject.general_info.aStabilityInner
            * dike_traject.general_info.TrajectLength
            / dike_traject.general_info.bStabilityInner
        )
        n_overflow = 1
        n_revetment = 3
        omegaRevetment = 0.1

        _beta_cs_piping = pf_to_beta(
            dike_traject.general_info.Pmax
            * dike_traject.general_info.omegaPiping
            / n_piping
        )
        _beta_cs_revetment = pf_to_beta(
            dike_traject.general_info.Pmax * omegaRevetment / n_revetment
        )
        _beta_cs_stabinner = pf_to_beta(
            dike_traject.general_info.Pmax
            * dike_traject.general_info.omegaStabilityInner
            / n_stab
        )
        _beta_cs_overflow = pf_to_beta(
            dike_traject.general_info.Pmax
            * dike_traject.general_info.omegaOverflow
            / n_overflow
        )
        return cls(
            beta_cs_piping=_beta_cs_piping,
            beta_cs_revetment=_beta_cs_revetment,
            beta_cs_stabinner=_beta_cs_stabinner,
            beta_cs_overflow=_beta_cs_overflow,
            dike_traject_b_piping=dike_traject.general_info.bPiping,
            dike_traject_b_stability_inner=dike_traject.general_info.bStabilityInner,
        )


class TargetReliabilityStrategy(StrategyProtocol):
    """Subclass for evaluation in accordance with basic OI2014 approach.
    This ensures that for a certain time horizon, each section satisfies the cross-sectional target reliability
    """

    def __init__(
        self, strategy_input: StrategyInputTargetReliability, config: VrtoolConfig
    ):
        # Mapping as in previuos `StrategyBase`
        self.discount_rate = config.discount_rate
        self.config = config
        self.OI_horizon = config.OI_horizon
        self.mechanisms = config.mechanisms
        self._time_periods = config.T
        self.LE_in_section = config.LE_in_section

        # New mappings
        # self.options = strategy_input.options
        self._section_as_input_dict = strategy_input.section_as_input_dict

    def get_total_lcc_and_risk(self, step_number: int) -> tuple[float, float]:
        return float("nan"), float("nan")

    @staticmethod
    def _id_to_name(found_id: str, measure_table: pd.DataFrame):
        """
        Previously in tools. Only used once within this evaluate method.
        """
        return measure_table.loc[measure_table["ID"].astype(str) == str(found_id)][
            "Name"
        ].values[0]

    def _get_beta_t_dictionary(
        self,
        cross_sectional_requirements: CrossSectionalRequirements,
        dike_section_length: float,
    ) -> dict[str, float]:
        # convert beta_cs to beta_section in order to correctly search self.options[section]
        # TODO THIS IS CURRENTLY INCONSISTENT WITH THE WAY IT IS CALCULATED: it should be coupled to whether the length effect within sections is turned on or not
        if self.LE_in_section:
            logging.warning(
                "In evaluate for TargetReliabilityStrategy: THIS CODE ON LENGTH EFFECT WITHIN SECTIONS SHOULD BE TESTED"
            )

        return {
            MechanismEnum.PIPING.name: cross_sectional_requirements.calculate_beta_t_piping(
                dike_section_length, self.LE_in_section
            ),
            MechanismEnum.STABILITY_INNER.name: cross_sectional_requirements.calculate_beta_t_sabinner(
                dike_section_length, self.LE_in_section
            ),
            MechanismEnum.OVERFLOW.name: cross_sectional_requirements.beta_cs_overflow,
            MechanismEnum.REVETMENT.name: cross_sectional_requirements.beta_cs_revetment,
        }

    def evaluate(
        self,
        dike_traject: DikeTraject,
        splitparams: bool = False,
    ):
        # Previous approach instead of self._time_periods = config.T:
        # _first_section_solution = solutions_dict[list(solutions_dict.keys())[0]]
        # cols = list(_first_section_solution.MeasureData["Section"].columns.values)

        # Rank sections based on 2075 Section probability
        beta_horizon = []
        for _dike_section in dike_traject.sections:
            beta_horizon.append(
                _dike_section.section_reliability.SectionReliability.loc["Section"][
                    self.OI_horizon
                ]
            )

        section_indices = np.argsort(beta_horizon)
        measure_cols = ["Section", "option_index", "LCC", "BC"]

        if splitparams:
            _taken_measures = pd.DataFrame(
                data=[
                    [
                        None,
                        None,
                        0,
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                    ]
                ],
                columns=measure_cols
                + [
                    "ID",
                    "name",
                    "year",
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
        _base_traject_probability = make_traject_df(dike_traject, self._time_periods)
        _probability_steps = [copy.deepcopy(_base_traject_probability)]
        _traject_probability = copy.deepcopy(_base_traject_probability)

        _cross_sectional_requirements = CrossSectionalRequirements.from_dike_traject(
            dike_traject
        )
        for j in section_indices:
            _dike_section = dike_traject.sections[j]
            _beta_t = self._get_beta_t_dictionary(
                _cross_sectional_requirements, _dike_section.Length
            )
            # find cheapest design that satisfies betatcs in 50 years from invest year
            # previously _selected_section_as_input = self.options[_dike_section.name]
            _selected_section_as_input = self._section_as_input_dict[_dike_section.name]
            _invest_year = _selected_section_as_input.min_year
            _target_year = _invest_year + 50

            # make PossibleMeasures dataframe
            _possible_measures = copy.deepcopy(_selected_section_as_input)
            # filter for mechanisms that are considered
            for mechanism in dike_traject.mechanisms:
                _possible_measures = _possible_measures.loc[
                    self.options[_dike_section.name][(mechanism.name, _target_year)]
                    > _beta_t[mechanism.name]
                ]

            if not any(_possible_measures):
                # continue to next section if weakest has no more measures
                logging.warning(
                    "Warning: for Target reliability strategy no suitable measures were found for section {}".format(
                        _dike_section.name
                    )
                )
                continue
            # calculate LCC
            _lcc = calc_tc(
                _possible_measures,
                self.discount_rate,
                horizon=_selected_section_as_input[MechanismEnum.OVERFLOW.name].columns[
                    -1
                ],
            )

            # select measure with lowest cost
            idx = np.argmin(_lcc)

            measure = _possible_measures.iloc[idx]
            option_index = _possible_measures.index[idx]
            # calculate achieved risk reduction & BC ratio compared to base situation
            _r_base, _dr, _t_r = calc_tr(
                _dike_section.name,
                measure,
                _traject_probability,
                original_section=_traject_probability.loc[_dike_section.name],
                discount_rate=self.discount_rate,
                horizon=self._time_periods[-1],
                damage=dike_traject.general_info.FloodDamage,
            )
            _bc = _dr / _lcc[idx]

            if splitparams:
                _found_id = measure["ID"].values[0]
                # TODO: We don't have the names as they were anymore :/
                # solutions_dict[i.name].measure_table
                # Which should translate to something like  SectionAsInput.get_measure_name_by_id()
                name = self._id_to_name(_found_id, self._section_as_input_dict)
                data_opt = pd.DataFrame(
                    [
                        [
                            _dike_section.name,
                            option_index,
                            _lcc[idx],
                            _bc,
                            measure["ID"].values[0],
                            name,
                            measure["year"].values[0],
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
                            _dike_section.name,
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
                _dike_section.name, _traject_probability, measure
            )
            _probability_steps.append(copy.deepcopy(_traject_probability))
        self.TakenMeasures = _taken_measures
        self.Probabilities = _probability_steps
