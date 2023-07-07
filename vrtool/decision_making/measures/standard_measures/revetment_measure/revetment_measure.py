import copy
import math

from math import isnan
import numpy as np
from scipy.interpolate import interp1d

from vrtool.common.dike_traject_info import DikeTrajectInfo
from vrtool.decision_making.measures.measure_protocol import MeasureProtocol
from vrtool.decision_making.measures.standard_measures.revetment_measure.revetment_measure_data_evaluator import (
    RevetmentMeasureDataBuilder,
)

from vrtool.decision_making.measures.standard_measures.revetment_measure.revetment_measure_result import (
    RevetmentMeasureResult,
)
from vrtool.decision_making.measures.standard_measures.revetment_measure.revetment_measure_result_collection import (
    RevetmentMeasureResultCollection,
)
from vrtool.failure_mechanisms.revetment.revetment_calculator import RevetmentCalculator
from vrtool.failure_mechanisms.revetment.revetment_data_class import RevetmentDataClass

from vrtool.flood_defence_system.dike_section import DikeSection
from vrtool.flood_defence_system.mechanism_reliability_collection import (
    MechanismReliabilityCollection,
)
from vrtool.flood_defence_system.section_reliability import SectionReliability
from vrtool.probabilistic_tools.probabilistic_functions import pf_to_beta


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
        # TODO (VRTOOL-187).
        # This class has been introduced to deal with the output to solutions.to_dataframe.
        # Consider removing it (or integrating it) in case SectionReliability can host these matrices.
        _results_collection = RevetmentMeasureResultCollection()
        _results_collection.measure_id = self.parameters["ID"]
        _results_collection.measure_name = self.parameters["Name"]
        _results_collection.reinforcement_type = self.parameters["Type"]
        _results_collection.combinable_type = self.parameters["Class"]
        _results_collection.revetment_measure_results = self._get_intermediate_measures(
            dike_section, _revetment, _beta_targets, _transition_levels, self.config.T
        )

        # 5. Return Beta and Cost matrices
        self.measures = _results_collection
        # self._get_configured_section_reliability(
        #     dike_section, traject_info, _results_collection
        # )

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

        for _beta_target in beta_targets:
            for _transition_level in transition_levels:
                _measures_per_year = []
                for _measure_year in revetment_years:
                    _measures_per_year.append(
                        self._get_measure_per_year(
                            dike_section,
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

    def _get_measure_per_year(
        self,
        dike_section: DikeSection,
        revetment: RevetmentDataClass,
        beta_target: float,
        transition_level: float,
        measure_year: int,
    ):
        # 3.1. Get measure Beta and cost per year.
        _revetment_measures_collection = (
            RevetmentMeasureDataBuilder().build_revetment_measure_data_collection(
                dike_section.crest_height,
                revetment,
                beta_target,
                transition_level,
                measure_year,
            )
        )
        _stone_beta_list, _grass_beta_list = zip(
            *(
                (rm.beta_block_revetment, rm.beta_grass_revetment)
                for rm in _revetment_measures_collection
            )
        )
        # Get the simple grass beta.
        _grass_beta = self._get_grass_revetment_beta_from_vector(_grass_beta_list)
        _combined_beta = RevetmentCalculator.calculate_combined_beta(
            _stone_beta_list, _grass_beta
        )
        _cost = sum(
            map(
                lambda x: x.get_total_cost(dike_section.Length),
                _revetment_measures_collection,
            )
        )
        return RevetmentMeasureResult(
            year=measure_year,
            beta_target=beta_target,
            beta_combined=_combined_beta,
            transition_level=transition_level,
            cost=_cost,
            revetment_measures=_revetment_measures_collection,
        )

    def _get_interpolated_measures(
        self,
        available_measures: list[RevetmentMeasureResult],
        config_years: list[int],
        revetment_years: list[int],
    ) -> list[RevetmentMeasureResult]:
        _diff_revetment_years = revetment_years[0] - config_years[0]
        _corrected_revetment_years = [
            ry - _diff_revetment_years for ry in revetment_years
        ]

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

    def _get_configured_section_reliability(
        self,
        dike_section: DikeSection,
        traject_info: DikeTrajectInfo,
        results_collection: RevetmentMeasureResultCollection,
    ) -> SectionReliability:
        section_reliability = SectionReliability()

        mechanism_names = (
            dike_section.section_reliability.failure_mechanisms.get_available_mechanisms()
        )
        for mechanism_name in mechanism_names:
            calc_type = dike_section.mechanism_data[mechanism_name][0][1]
            mechanism_reliability_collection = (
                self._get_configured_mechanism_reliability_collection(
                    mechanism_name,
                    calc_type,
                    dike_section,
                    traject_info,
                    results_collection,
                )
            )
            section_reliability.failure_mechanisms.add_failure_mechanism_reliability_collection(
                mechanism_reliability_collection
            )

        return section_reliability

    def _get_configured_mechanism_reliability_collection(
        self,
        mechanism_name: str,
        calc_type: str,
        dike_section: DikeSection,
        traject_info: DikeTrajectInfo,
        results_collection: RevetmentMeasureResultCollection,
    ) -> MechanismReliabilityCollection:
        mechanism_reliability_collection = MechanismReliabilityCollection(
            mechanism_name, calc_type, self.config.T, self.config.t_0, 0
        )

        for year_to_calculate in mechanism_reliability_collection.Reliability.keys():
            mechanism_reliability_collection.Reliability[
                year_to_calculate
            ].Input = copy.deepcopy(
                dike_section.section_reliability.failure_mechanisms.get_mechanism_reliability_collection(
                    mechanism_name
                )
                .Reliability[year_to_calculate]
                .Input
            )

            mechanism_reliability = mechanism_reliability_collection.Reliability[
                year_to_calculate
            ]
            # mechanism_reliability["revetment_input"] =

        mechanism_reliability_collection.generate_LCR_profile(
            dike_section.section_reliability.load,
            traject_info=traject_info,
        )

        return mechanism_reliability_collection
