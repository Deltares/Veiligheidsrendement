import copy
from itertools import groupby

from math import isnan
import numpy as np
from scipy.interpolate import interp1d

from vrtool.common.dike_traject_info import DikeTrajectInfo
from vrtool.common.hydraulic_loads.load_input import LoadInput
from vrtool.decision_making.measures.measure_protocol import MeasureProtocol
from vrtool.decision_making.measures.standard_measures.revetment_measure.revetment_measure_result_builder import (
    RevetmentMeasureResultBuilder,
)

from vrtool.decision_making.measures.standard_measures.revetment_measure.revetment_measure_result import (
    RevetmentMeasureResult,
)
from vrtool.decision_making.measures.standard_measures.revetment_measure.revetment_measure_result_collection import (
    RevetmentMeasureSectionReliability,
    RevetmentMeasureResultCollection,
)
from vrtool.failure_mechanisms.mechanism_input import MechanismInput
from vrtool.failure_mechanisms.revetment.revetment_calculator import RevetmentCalculator
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

    def _get_min_beta_target(self, dike_section: DikeSection) -> float:
        return float(
            max(
                map(
                    lambda x: x.Beta,
                    dike_section.section_reliability.failure_mechanisms.get_mechanism_reliability_collection(
                        "Revetment"
                    ).Reliability.values(),
                )
            )
        )

    def _get_beta_target_vector(self, min_beta: float, p_max: float) -> list[float]:
        _max_beta = pf_to_beta(p_max / self.max_pf_factor_block)
        _step = (_max_beta - min_beta) / self.n_steps_block
        return list(np.arange(min_beta, _max_beta, _step))

    def _get_transition_level_vector(
        self, current_transition_level: float, crest_height: float
    ) -> list[float]:
        return list(
            np.arange(
                current_transition_level,
                crest_height,
                self.transition_level_increase_step,
            )
        )

    def _get_revetment(self, dike_section: DikeSection) -> RevetmentDataClass:
        _reliability_collection = dike_section.section_reliability.failure_mechanisms.get_mechanism_reliability_collection(
            "Revetment"
        )
        _first_year = list(_reliability_collection.Reliability.keys())[0]
        return _reliability_collection.Reliability[_first_year].Input.input[
            "revetment_input"
        ]

    def evaluate_measure(
        self,
        dike_section: DikeSection,
        traject_info: DikeTrajectInfo,
        preserve_slope: bool,
    ):
        _revetment = self._get_revetment(dike_section)

        # 1. Get beta targets.
        # TODO. Currently only one beta target available due to the step = 4.
        _beta_targets = self._get_beta_target_vector(
            self._get_min_beta_target(dike_section), traject_info.Pmax
        )

        # 2. Get transition levels.
        _transition_levels = self._get_transition_level_vector(
            _revetment.current_transition_level, dike_section.crest_height
        )

        # 3. Iterate over beta_targets - transition level - year.
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
                    self.parameters["Type"],
                    self.parameters["Type"],
                    dike_section,
                    transition_grouping,
                )
                self.measures.beta_target_results.append(_beta_target_results)

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
        beta_targets: list[float],
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

    def _get_grass_revetment_beta_from_vector(
        self, grass_revetment_betas: list[float]
    ) -> float:
        # The `grass_revetment_betas` contain a list where only the last values are valid.
        # At the same time, these last values are always the same.
        def filter_valid_revetment(revetment_beta: float) -> bool:
            return not isnan(revetment_beta)

        return float(list(filter(filter_valid_revetment, grass_revetment_betas))[0])

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
                revetment_measures=[
                    rm.revetment_measures for rm in available_measures
                ],  # TODO: Not very happy about this type.
            )
            _interpolated_measures.append(_interpolated_measure)
        return _interpolated_measures

    def _get_configured_section_reliability_and_cost(
        self,
        mechanism_name: str,
        calc_type: str,
        dike_section: DikeSection,
        revetment_measure_results: list[RevetmentMeasureResult],
    ) -> tuple[SectionReliability, float]:

        section_reliability = SectionReliability()
        _failure_mechanism_collection = copy.deepcopy(
            dike_section.section_reliability.failure_mechanisms
        )
        _assessment_revetment = (
            _failure_mechanism_collection.get_mechanism_reliability_collection(
                mechanism_name
            )
        )
        _assessment_revetment.Reliability = (
            self._get_mechanism_reliabilty_for_beta_transition(
                mechanism_name, calc_type, revetment_measure_results
            )
        )
        section_reliability.failure_mechanisms = _failure_mechanism_collection
        # TODO (VRTOOL-187)
        # Is this really required for revetments? Should we not better have our own RevetmentSectionReliability?
        section_reliability.calculate_section_reliability()
        return section_reliability, sum([r.cost for r in revetment_measure_results])

    def _get_mechanism_reliabilty_for_beta_transition(
        self,
        mechanism_name: str,
        calc_type: str,
        revetment_measure_results: list[RevetmentMeasureResult],
    ) -> dict[str, MechanismReliability]:
        class RevetmentMeasureMechanismReliability(MechanismReliability):
            def calculate_reliability(
                self,
                strength: MechanismInput,
                load: LoadInput,
                mechanism: str,
                year: float,
                traject_info: DikeTrajectInfo,
            ):
                # TODO (VRTOOL-187).
                # This is done to prevent a Revetment mechanism to be calculated because we already know its beta combined.
                pass

        _reliability_dict = {}
        for result in revetment_measure_results:
            mechanism_reliability = RevetmentMeasureMechanismReliability(
                mechanism_name, calc_type, self.config.t_0
            )
            mechanism_reliability.Beta = result.beta_combined
            mechanism_reliability.Pf = beta_to_pf(result.beta_combined)
            _reliability_dict[str(result.year)] = mechanism_reliability
        return _reliability_dict
