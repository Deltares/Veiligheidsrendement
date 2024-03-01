import copy
import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from vrtool.common.enums.mechanism_enum import MechanismEnum
from vrtool.decision_making.strategies.strategy_protocol import StrategyProtocol
from vrtool.decision_making.strategy_evaluation import (
    calc_tc,
    calc_tr,
    implement_option,
    make_traject_df,
    compute_total_risk,
)
from vrtool.defaults.vrtool_config import VrtoolConfig
from vrtool.flood_defence_system.dike_traject import DikeTraject
from vrtool.optimization.measures.section_as_input import SectionAsInput
from vrtool.optimization.strategy_input.strategy_input_target_reliability import (
    StrategyInputTargetReliability,
)
from vrtool.probabilistic_tools.probabilistic_functions import pf_to_beta
from vrtool.optimization.measures.aggregated_measures_combination import AggregatedMeasureCombination


@dataclass
class CrossSectionalRequirements:
    pf_cs: dict[MechanismEnum, np.ndarray]

    dike_traject_b_piping: float
    dike_traject_b_stability_inner: float

    @property
    def beta_cs_piping(self) -> np.ndarray:
        return pf_to_beta(self.pf_cs[MechanismEnum.PIPING])

    @property
    def beta_cs_stabinner(self) -> np.ndarray:
        return pf_to_beta(self.pf_cs[MechanismEnum.STABILITY_INNER])

    def calculate_beta_t_piping(
        self, dike_section_length: float, le_in_section: bool
    ) -> np.ndarray:
        if not le_in_section:
            return pf_to_beta(self.pf_cs[MechanismEnum.PIPING])
        return pf_to_beta(
            self.pf_cs[MechanismEnum.PIPING] * (dike_section_length / self.dike_traject_b_piping)
        )

    def calculate_beta_t_stabinner(
        self, dike_section_length: float, le_in_section: bool
    ) -> np.ndarray:
        if not le_in_section:
            return pf_to_beta(self.pf_cs[MechanismEnum.STABILITY_INNER])
        return pf_to_beta(
            self.pf_cs[MechanismEnum.STABILITY_INNER]
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

        _pf_cs_piping = (
            dike_traject.general_info.Pmax
            * dike_traject.general_info.omegaPiping
            / n_piping
        )
        _pf_cs_revetment = (
            dike_traject.general_info.Pmax * omegaRevetment / n_revetment
        )
        _pf_cs_stabinner = (
            dike_traject.general_info.Pmax
            * dike_traject.general_info.omegaStabilityInner
            / n_stab
        )
        _pf_cs_overflow = (
            dike_traject.general_info.Pmax
            * dike_traject.general_info.omegaOverflow
            / n_overflow
        )
        return cls(
            pf_cs = {MechanismEnum.PIPING: _pf_cs_piping,
                     MechanismEnum.STABILITY_INNER: _pf_cs_stabinner,
                     MechanismEnum.OVERFLOW: _pf_cs_overflow,
                     MechanismEnum.REVETMENT: _pf_cs_revetment},
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
        # Necessary config parameters:
        self.OI_horizon = config.OI_horizon

        # New mappings
        self.Pf = strategy_input.Pf
        self.D = strategy_input.D
        self.sections = strategy_input.sections

    def get_total_lcc_and_risk(self, step_number: int) -> tuple[float, float]:
        return float("nan"), float("nan")

    def evaluate(
        self,
        dike_traject: DikeTraject,
    ):
        # Previous approach instead of self._time_periods = config.T:
        # _first_section_solution = solutions_dict[list(solutions_dict.keys())[0]]
        # cols = list(_first_section_solution.MeasureData["Section"].columns.values)
        def _check_cross_sectional_requirements(measure: AggregatedMeasureCombination, cross_sectional_requirements, year, mechanisms
                                                )-> bool:
            """This function checks if the cross-sectional requirements are met for a given measure and year.
            If the requirements are not met for any of the mechanisms, the function returns False, otherwise True."""
            for mechanism in mechanisms:
                if mechanism in [MechanismEnum.OVERFLOW, MechanismEnum.REVETMENT]:
                    #look in sh, if any mechanism is not satisfied, return a False
                    if measure.sh_combination.mechanism_year_collection.get_probability(mechanism,year) > cross_sectional_requirements.pf_cs[mechanism]:
                        return False
                elif mechanism in [MechanismEnum.PIPING, MechanismEnum.STABILITY_INNER]:    
                    if measure.sg_combination.mechanism_year_collection.get_probability(mechanism,year) > cross_sectional_requirements.pf_cs[mechanism]:
                        return False
            return True
        
        # Get initial failure probabilities at design horizon. #TODO think about what year is to be used here.
        initial_section_pfs = [section.initial_assessment.get_section_probability(self.OI_horizon) for section in self.sections]

        # Rank sections based on initial probability
        section_order = np.flip(np.argsort(initial_section_pfs))

        # get the cross-sectional requirements for the dike traject (probability)
        _cross_sectional_requirements = CrossSectionalRequirements.from_dike_traject(
            dike_traject
        )
        #and the risk for each step
        _taken_measures = {}
        _taken_measures_indices = []
        for _section_idx in section_order:
            # add probability for this step:

            #get the first possible investment year from the aggregated measures
            _section_as_input = self.sections[_section_idx]
            _invest_year = min([measure.year for measure in _section_as_input.aggregated_measure_combinations])
            _design_horizon_year = _invest_year + self.OI_horizon
            #for each aggregate measure check if the cross-sectional requirements are met
            _satisfied_bool = [_check_cross_sectional_requirements(_measure, _cross_sectional_requirements, _design_horizon_year, _section_as_input.mechanisms) for _measure in _section_as_input.aggregated_measure_combinations]
            #generate bool for each measure with year in investment year
            _valid_year_bool = [measure.year == _invest_year for measure in _section_as_input.aggregated_measure_combinations]

            #get the measure with the lowest lcc that satisfies both _satisfied_bool and _valid_year_bool
            _valid_measures = [_measure for _measure, _satisfied, _valid_year in zip(_section_as_input.aggregated_measure_combinations, _satisfied_bool, _valid_year_bool) if _satisfied and _valid_year]
            if len(_valid_measures) == 0:
                #if no measures satisfy the requirements, continue to the next section
                logging.warning(
                                        "Geen maatregelen gevonden die voldoen aan doorsnede-eisen op dijkvak {}. Er wordt geen maatregel uitgevoerd.".format(
                        _section_as_input.section_name
                    )
                )
                continue
            #get measure with lowest lcc from _valid_measures
            _lcc = [measure.lcc for measure in _valid_measures]
            idx = np.argmin(_lcc)
            _taken_measures[self.sections[_section_idx].section_name] = _valid_measures[idx]
            measure_idx = self.sections[_section_idx].get_combination_idx_for_aggregate(_taken_measures[self.sections[_section_idx].section_name])
            _taken_measures_indices.append((_section_idx, measure_idx[0]+1, measure_idx[1]+1))


        # For output we need to give the list of measure indices, the total_risk per step, and the probabilities per step
        # First we get, and update the probabilities per step
        #we need to track probability for each step
        init_probability = {mech: self.Pf[mech][:,0,:] for mech in self.Pf.keys()}
        self.probabilities_per_step = [copy.deepcopy(init_probability)]
        self.total_risk_per_step = [compute_total_risk(self.probabilities_per_step[-1], self.D)]

        for step in range(0, len(_taken_measures)):
            section_id = _taken_measures_indices[step][0]
            self.probabilities_per_step.append(copy.deepcopy(self.probabilities_per_step[-1]))
            self.probabilities_per_step[-1] = implement_option(
                self.probabilities_per_step[-1], 
                _taken_measures_indices[step], 
                _taken_measures[self.sections[section_id].section_name]
                )
            self.total_risk_per_step.append(compute_total_risk(self.probabilities_per_step[-1], self.D))
        
        self.measures_taken = _taken_measures_indices
