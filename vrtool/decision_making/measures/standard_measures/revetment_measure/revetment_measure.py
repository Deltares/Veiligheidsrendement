import copy
import math
from itertools import groupby

import numpy as np
from scipy.interpolate import interp1d

from vrtool.common.dike_traject_info import DikeTrajectInfo
from vrtool.common.enums.computation_type_enum import ComputationTypeEnum
from vrtool.common.enums.mechanism_enum import MechanismEnum
from vrtool.common.hydraulic_loads.load_input import LoadInput
from vrtool.decision_making.measures.measure_protocol import MeasureProtocol
from vrtool.decision_making.measures.standard_measures.revetment_measure.revetment_measure_result import (
    RevetmentMeasureResult,
)
from vrtool.decision_making.measures.standard_measures.revetment_measure.revetment_measure_result_builder import (
    RevetmentMeasureResultBuilder,
)
from vrtool.decision_making.measures.standard_measures.revetment_measure.revetment_measure_result_collection import (
    RevetmentMeasureResultCollection,
    RevetmentMeasureSectionReliability,
)
from vrtool.failure_mechanisms.mechanism_input import MechanismInput
from vrtool.failure_mechanisms.revetment.revetment_data_class import RevetmentDataClass
from vrtool.flood_defence_system.dike_section import DikeSection
from vrtool.flood_defence_system.mechanism_reliability import MechanismReliability
from vrtool.flood_defence_system.section_reliability import SectionReliability
from vrtool.probabilistic_tools.probabilistic_functions import beta_to_pf, pf_to_beta


class RevetmentMeasure(MeasureProtocol):
    def __init__(self):
        self.parameters = {}
        self.measures = None

    @property
    def transition_level_increase_step(self) -> float:
        return self.parameters["transition_level_increase_step"]

    @property
    def max_pf_factor_block(self) -> float:
        return self.parameters["max_pf_factor_block"]

    @property
    def n_steps_block(self) -> int:
        return self.parameters["n_steps_block"]

    @property
    def minimal_stepsize(self) -> float:
        """
        step size for beta
        """
        return 0.3

    @property
    def margin_min_max_beta(self) -> float:
        """
        margin at which min_beta is seen as equal to max_beta.
        """
        return 0.1

    @property
    def tol_abs_beta_in_filtering(self) -> float:
        """
        absolute margin at which two betas are seen as equal in filtering.
        """
        return 1e-2

    @property
    def tol_rel_costs_in_filtering(self) -> float:
        """
        relative margin at which two costs are seen as equal to filtering.
        """
        return 1e-3

    def _get_beta_max(self, p_max: float) -> float:
        """
        get maximum beta for revetment

        Args:
            p_max (float): target probability for traject

        Returns:
            float: maximum beta
        """
        _max_beta = pf_to_beta(p_max / self.max_pf_factor_block)
        return _max_beta

    def _get_beta_target_vector(
        self, min_beta: float, max_beta: float
    ) -> np.ndarray[float]:
        """
        get a grid with beta values
        in principle n_step_block values, but stepsize is at most minimal_stepsize

        Args:
            min_beta (float): minimal beta value
            max_beta (float): maximum beta value

        Returns:
            np.ndarray[float]: list with grid values
        """
        _stepsize = (max_beta - min_beta) / (self.n_steps_block - 1)
        _minimal_nr_of_steps = 2
        if max_beta <= min_beta + self.margin_min_max_beta:
            _minimal_nr_of_steps = 1
        _nr_of_steps = 1 + round(
            (max_beta - min_beta) / max(self.minimal_stepsize, _stepsize)
        )
        _nr_of_steps = max(_minimal_nr_of_steps, _nr_of_steps)
        return np.linspace(min_beta, max_beta, _nr_of_steps)

    def _get_transition_level_vector(
        self,
        revetment: RevetmentDataClass,
        crest_height: float,
    ) -> list[float]:
        # Get max and current transition levels.
        _max_transition_level = revetment.get_transition_level_below_threshold(
            crest_height
        )
        _current_transition_level = revetment.current_transition_level

        # Check for a 'fast' exit.
        if _current_transition_level > crest_height:
            raise ValueError(
                "Transition level is higher than crest height. This is not allowed."
            )
        elif math.isclose(_current_transition_level, crest_height):
            return [_current_transition_level]

        # Current transition level is less than the crest height.
        # Get vector with transition levels.
        _level_vector = list(
            np.arange(
                _current_transition_level,
                crest_height,
                self.transition_level_increase_step,
            )
        )

        # Check we need to include the maximum transition level or not.
        # This happens when the step is just above it (VRTOOL-330).
        if _level_vector[-1] < _max_transition_level and (
            (_max_transition_level < crest_height)
            or (math.isclose(_max_transition_level, crest_height))
        ):
            _level_vector.append(_max_transition_level)
        return _level_vector

    def _get_revetment(self, dike_section: DikeSection) -> RevetmentDataClass:
        _reliability_collection = dike_section.section_reliability.failure_mechanisms.get_mechanism_reliability_collection(
            MechanismEnum.REVETMENT
        )
        _first_year = list(_reliability_collection.Reliability.keys())[0]
        return _reliability_collection.Reliability[_first_year].Input.input[
            "revetment_input"
        ]

    def _compare_costs(self, costs_first: float, costs_second: float) -> int:
        """
        compare two costs

        Returns:
            int: 0 if the costs are equal; +1 if the first is higher otherwise -1
        """
        if math.isclose(
            costs_first, costs_second, rel_tol=self.tol_abs_beta_in_filtering
        ):
            return 0
        elif costs_first > costs_second:
            return 1
        else:
            return -1

    def _compare_betas(self, beta_first: float, beta_second: float) -> int:
        """
        compare two betas

        Returns:
            int: 0 if the betas are equal; +1 if the first is higher otherwise -1
        """
        if math.isclose(
            beta_first, beta_second, abs_tol=self.tol_abs_beta_in_filtering
        ):
            return 0
        elif beta_first > beta_second:
            return 1
        else:
            return -1

    def _get_filtered_measures(self):
        """
        remove measures in measures.result_collection that have a worse cost-beta relation
        than any of the other measures
        also remove measures with (almost) the same cost and beta as an earlier measure
        """
        filtered_msrs = []
        for i, msr in enumerate(self.measures.result_collection):
            remove_msr = False
            if msr.cost > 0:
                for j, cmp in enumerate(self.measures.result_collection):
                    if i == j:
                        continue
                    beta_msr = msr.section_reliability.SectionReliability["0"][
                        MechanismEnum.REVETMENT.__str__()
                    ]
                    beta_cmp = cmp.section_reliability.SectionReliability["0"][
                        MechanismEnum.REVETMENT.__str__()
                    ]
                    cmp_betas = self._compare_betas(beta_msr, beta_cmp)
                    cmp_costs = self._compare_costs(msr.cost, cmp.cost)
                    if cmp_costs == 0 and cmp_betas == 0:
                        remove_msr = remove_msr or j < i
                    elif cmp_costs >= 0 and cmp_betas <= 0:
                        remove_msr = True
            if not remove_msr:
                filtered_msrs.append(msr)

        self.measures.result_collection = filtered_msrs

    def evaluate_measure(
        self,
        dike_section: DikeSection,
        traject_info: DikeTrajectInfo,
        preserve_slope: bool,
    ):
        _revetment = self._get_revetment(dike_section)

        # 1. Get beta targets.
        _min_beta = _revetment.beta_stone
        _max_beta = self._get_beta_max(traject_info.Pmax)
        _beta_targets = self._get_beta_target_vector(_min_beta, _max_beta)

        # 2. Get transition levels.
        _transition_levels = self._get_transition_level_vector(
            _revetment,
            dike_section.crest_height,
        )

        # 3 & 4. Iterate over beta_targets - transition level - year.
        _intermediate_measures = self._get_intermediate_measures(
            dike_section, _revetment, _beta_targets, _transition_levels, self.config.T
        )

        # 5. Return Beta and Cost matrices
        self.measures = RevetmentMeasureResultCollection()
        for beta_target, beta_grouping in self._get_grouped_intermediate_results(
            _intermediate_measures
        ).items():
            for transition_level, transition_grouping in beta_grouping.items():
                _beta_target_results = RevetmentMeasureSectionReliability()
                _beta_target_results.measure_id = self.parameters["ID"]
                _beta_target_results.measure_name = self.parameters["Name"]
                _beta_target_results.reinforcement_type = self.parameters["Type"]
                _beta_target_results.combinable_type = self.parameters["Class"]
                _beta_target_results.measure_year = self.parameters["year"]
                _beta_target_results.beta_target = beta_target
                _beta_target_results.transition_level = transition_level
                _beta_target_results.revetment_measure_results = transition_grouping
                (
                    _beta_target_results.section_reliability,
                    _beta_target_results.cost,
                ) = self._get_configured_section_reliability_and_cost(
                    MechanismEnum.get_enum(self.parameters["Type"]),
                    self.parameters["Type"],
                    dike_section,
                    transition_grouping,
                )
                self.measures.result_collection.append(_beta_target_results)

        self._get_filtered_measures()

    def _get_grouped_intermediate_results(
        self, ungrouped_measures: list[RevetmentMeasureResult]
    ) -> dict:
        def get_sorted(
            measures: list[RevetmentMeasureResult], lambda_expression
        ) -> dict:
            _sorted_list = sorted(measures, key=lambda_expression)
            return groupby(_sorted_list, key=lambda_expression)

        _results_dict = {}
        for _beta_key, _beta_group in get_sorted(
            ungrouped_measures, lambda x: x.beta_target
        ):
            _results_dict[_beta_key] = {}
            for _transition_key, _transition_group in get_sorted(
                _beta_group, lambda x: x.transition_level
            ):
                _results_dict[_beta_key][_transition_key] = list(_transition_group)
        return _results_dict

    def _get_intermediate_measures(
        self,
        dike_section: DikeSection,
        revetment: RevetmentDataClass,
        beta_targets: np.ndarray[float],
        transition_levels: list[float],
        config_years: list[int],
    ) -> list[RevetmentMeasureResult]:
        _intermediate_measures = []
        revetment_years = revetment.get_available_years()
        _result_builder = RevetmentMeasureResultBuilder()
        for _beta_target in beta_targets:
            for _transition_level in transition_levels:
                _measures_per_year = []
                for _measure_year in revetment_years:
                    _measures_per_year.append(
                        _result_builder.build(
                            dike_section.crest_height,
                            dike_section.Length,
                            revetment,
                            _beta_target,
                            _transition_level,
                            _measure_year,
                            self.unit_costs,
                        )
                    )
                # 4. Interpolate with years to calculate.
                _intermediate_measures.extend(
                    self._get_interpolated_measures(
                        _measures_per_year,
                        config_years,
                        revetment_years,
                    )
                )

        return _intermediate_measures

    def _get_interpolated_measures(
        self,
        available_measures: list[RevetmentMeasureResult],
        config_years: list[int],
        revetment_years: list[int],
    ) -> list[RevetmentMeasureResult]:
        _corrected_revetment_years = [ry - self.t_0 for ry in revetment_years]

        def _interpolate(values_to_interpolate: list[float], year: int) -> float:
            return float(
                interp1d(
                    _corrected_revetment_years,
                    values_to_interpolate,
                    fill_value=("extrapolate"),
                )(year)
            )

        _interpolated_measures = []
        _sample = available_measures[-1]
        for _year in config_years:
            _interpolated_beta = _interpolate(
                [am.beta_combined for am in available_measures], _year
            )
            _interpolated_cost = _interpolate(
                [am.cost for am in available_measures], _year
            )
            _interpolated_measure = RevetmentMeasureResult(
                year=_year,
                beta_target=_sample.beta_target,
                transition_level=_sample.transition_level,
                beta_combined=_interpolated_beta,
                cost=_interpolated_cost,
            )
            _interpolated_measures.append(_interpolated_measure)
        return _interpolated_measures

    def _get_configured_section_reliability_and_cost(
        self,
        mechanism: MechanismEnum,
        calc_type: ComputationTypeEnum,
        dike_section: DikeSection,
        revetment_measure_results: list[RevetmentMeasureResult],
    ) -> tuple[SectionReliability, float]:

        section_reliability = SectionReliability()
        _failure_mechanism_collection = copy.deepcopy(
            dike_section.section_reliability.failure_mechanisms
        )
        _assessment_revetment = (
            _failure_mechanism_collection.get_mechanism_reliability_collection(
                mechanism
            )
        )
        _assessment_revetment.Reliability = (
            self._get_mechanism_reliabilty_for_beta_transition(
                mechanism, calc_type, revetment_measure_results
            )
        )
        section_reliability.failure_mechanisms = _failure_mechanism_collection
        # TODO (VRTOOL-187)
        # Is this really required for revetments? Should we not better have our own RevetmentSectionReliability?
        section_reliability.calculate_section_reliability(
            dike_section.get_cross_sectional_properties()
        )
        return section_reliability, revetment_measure_results[0].cost

    def _get_mechanism_reliabilty_for_beta_transition(
        self,
        mechanism: MechanismEnum,
        calc_type: ComputationTypeEnum,
        revetment_measure_results: list[RevetmentMeasureResult],
    ) -> dict[MechanismEnum, MechanismReliability]:
        class RevetmentMeasureMechanismReliability(MechanismReliability):
            def calculate_reliability(
                self,
                strength: MechanismInput,
                load: LoadInput,
                mechanism: MechanismEnum,
                year: float,
                traject_info: DikeTrajectInfo,
            ):
                # TODO (VRTOOL-187).
                # This is done to prevent a Revetment mechanism to be calculated because we already know its beta combined.
                pass

        _reliability_dict = {}
        for result in revetment_measure_results:
            mechanism_reliability = RevetmentMeasureMechanismReliability(
                mechanism, calc_type, self.config.t_0
            )
            mechanism_reliability.Beta = result.beta_combined
            mechanism_reliability.Pf = beta_to_pf(result.beta_combined)
            _reliability_dict[str(result.year)] = mechanism_reliability
        return _reliability_dict
