import logging
import math


import numpy as np
from scipy.interpolate import interp1d

from vrtool.common.dike_traject_info import DikeTrajectInfo
from vrtool.decision_making.measures.measure_protocol import MeasureProtocol
from vrtool.decision_making.measures.standard_measures.revetment_measure.revetment_measure_data import (
    RevetmentMeasureData,
)
from vrtool.decision_making.measures.standard_measures.revetment_measure.revetment_measure_result import (
    RevetmentMeasureResult,
)
from vrtool.decision_making.measures.standard_measures.revetment_measure.revetment_measure_result_collection import (
    RevetmentMeasureResultCollection,
)
from vrtool.failure_mechanisms.revetment.relation_stone_revetment import (
    RelationStoneRevetment,
)
from vrtool.failure_mechanisms.revetment.revetment_calculator import RevetmentCalculator
from vrtool.failure_mechanisms.revetment.revetment_data_class import RevetmentDataClass
from vrtool.failure_mechanisms.revetment.slope_part import (
    GrassSlopePart,
    SlopePartProtocol,
    StoneSlopePart,
)
from vrtool.failure_mechanisms.revetment.slope_part.grass_slope_part import GRASS_TYPE
from vrtool.flood_defence_system.dike_section import DikeSection
from vrtool.probabilistic_tools.probabilistic_functions import pf_to_beta


def bisection(f, a, b, tol):
    # approximates a root, R, of f bounded
    # by a and b to within tolerance
    # | f(m) | < tol with m the midpoint
    # between a and b Recursive implementation

    # check if a and b bound a root
    if np.sign(f(a)) == np.sign(f(b)):
        raise Exception("The scalars a and b do not bound a root")

    # get midpoint
    m = (a + b) / 2

    if np.abs(f(m)) < tol:
        # stopping condition, report m as root
        return m
    elif np.sign(f(a)) == np.sign(f(m)):
        # case where m is an improvement on a.
        # Make recursive call with a = m
        return bisection(f, m, b, tol)
    elif np.sign(f(b)) == np.sign(f(m)):
        # case where m is an improvement on b.
        # Make recursive call with b = m
        return bisection(f, a, m, tol)


class RevetmentMeasure(MeasureProtocol):
    @property
    def transition_level_increase_step(self) -> float:
        return self.parameters["transition_level_increase_step"]

    @property
    def max_pf_factor_block(self) -> float:
        return self.parameters["max_pf_factor_block"]

    @property
    def n_steps_block(self) -> int:
        return self.parameters["n_steps_block"]

    def _correct_revetment_measure_data(
        self,
        revetment_measures: list[RevetmentMeasureData],
        current_transition_level: float,
    ) -> RevetmentMeasureData:

        _stone_revetments = [
            rm
            for rm in revetment_measures
            if StoneSlopePart.is_stone_slope_part(rm.top_layer_type)
        ]
        if not _stone_revetments:
            raise ValueError("No stone revetment measure was found.")

        # TODO: Check whether I'm getting the last one or the first one.
        _last_stone_revetment = _stone_revetments[-1]

        for _revetment_measure in revetment_measures:
            if (
                GrassSlopePart.is_grass_part(_revetment_measure.top_layer_type)
                and _revetment_measure.begin_part < current_transition_level
            ):
                _revetment_measure.top_layer_thickness = (
                    _last_stone_revetment.top_layer_thickness
                )
                _revetment_measure.top_layer_type = 2026.0
                _revetment_measure.beta_block_revetment = (
                    _last_stone_revetment.beta_block_revetment
                )
                _revetment_measure.reinforce = True

    def _get_design_stone(
        self,
        calculated_beta: float,
        slope_part: SlopePartProtocol,
        stone_revetment: RelationStoneRevetment,
        evaluation_year: int,
    ):
        _recalculated_beta = float("nan")
        _is_reinforced = False

        if not StoneSlopePart.is_stone_slope_part(slope_part.top_layer_type):
            return slope_part.top_layer_thickness, _recalculated_beta, _is_reinforced

        _top_layer_thickness_collection, _revetment_betas_collection = zip(
            *(
                (spr.top_layer_thickness, spr.beta)
                for spr in slope_part.slope_part_relations
                if spr.year == stone_revetment.year
            )
        )
        stone_interpolation = interp1d(
            _top_layer_thickness_collection,
            np.array(_revetment_betas_collection) - calculated_beta,
            fill_value=("extrapolate"),
        )

        try:
            _top_layer_thickness = bisection(stone_interpolation, 0.0, 1.0, 0.01)
            _top_layer_thickness = math.ceil(_top_layer_thickness / 0.05) * 0.05
            _recalculated_beta = calculated_beta
            _is_reinforced = True
        except:
            _top_layer_thickness = float("nan")

        if _top_layer_thickness <= slope_part.top_layer_thickness:
            logging.warning("Design D is <= than the current D")
            _top_layer_thickness = slope_part.top_layer_thickness
            _recalculated_beta = float(
                self._evaluate_stone_revetment_data(slope_part, evaluation_year)
            )

        return _top_layer_thickness, _recalculated_beta, _is_reinforced

    def _get_stone_revetment_measure_data(
        self,
        slope_part: SlopePartProtocol,
        stone_revetment: RelationStoneRevetment,
        target_beta: float,
        evaluation_year: int,
    ) -> RevetmentMeasureData:
        (
            _new_top_layer_thickness,
            _new_beta_block_revetment,
            _is_reinforced,
        ) = self._get_design_stone(
            target_beta, slope_part, stone_revetment, evaluation_year
        )
        return RevetmentMeasureData(
            begin_part=slope_part.begin_part,
            end_part=slope_part.end_part,
            top_layer_type=slope_part.top_layer_type,
            previous_top_layer_type=slope_part.top_layer_type,
            top_layer_thickness=_new_top_layer_thickness,
            beta_block_revetment=_new_beta_block_revetment,
            beta_grass_revetment=float("nan"),
            reinforce=_is_reinforced,
            tan_alpha=slope_part.tan_alpha,
        )

    def _get_combined_revetment_data(
        self,
        slope_part: SlopePartProtocol,
        revetment_data: RevetmentDataClass,
        stone_revetment: RelationStoneRevetment,
        transition_level: float,
        evaluation_year: int,
        target_beta: float,
    ) -> tuple[RevetmentMeasureData, RevetmentMeasureData]:
        (
            _new_top_layer_thickness,
            _new_beta_block_revetment,
            _is_reinforced,
        ) = self._get_design_stone(
            target_beta, slope_part, stone_revetment, evaluation_year
        )
        _stone_rmd = RevetmentMeasureData(
            begin_part=slope_part.begin_part,
            end_part=transition_level,
            top_layer_type=slope_part.top_layer_type,
            previous_top_layer_type=slope_part.top_layer_type,
            top_layer_thickness=_new_top_layer_thickness,
            beta_block_revetment=_new_beta_block_revetment,
            beta_grass_revetment=float("nan"),
            reinforce=_is_reinforced,
            tan_alpha=slope_part.tan_alpha,
        )

        _grass_rmd = RevetmentMeasureData(
            begin_part=transition_level,
            end_part=slope_part.end_part,
            top_layer_type=GRASS_TYPE,
            previous_top_layer_type=slope_part.top_layer_type,
            top_layer_thickness=float("nan"),
            beta_block_revetment=float("nan"),
            beta_grass_revetment=self._evaluate_grass_revetment_data(
                evaluation_year, revetment_data
            ),
            reinforce=True,
            tan_alpha=slope_part.tan_alpha,
        )

        return _stone_rmd, _grass_rmd

    def _get_grass_revetment_data(
        self,
        slope_part: SlopePartProtocol,
        revetment_data: RevetmentDataClass,
        evaluation_year: int,
    ) -> RevetmentMeasureData:
        return RevetmentMeasureData(
            begin_part=slope_part.begin_part,
            end_part=slope_part.end_part,
            top_layer_type=GRASS_TYPE,
            previous_top_layer_type=slope_part.top_layer_type,
            top_layer_thickness=float("nan"),
            beta_block_revetment=float("nan"),
            beta_grass_revetment=self._evaluate_grass_revetment_data(
                evaluation_year, revetment_data
            ),
            reinforce=True,
            tan_alpha=slope_part.tan_alpha,
        )

    def _get_revetment_measure_data_collection(
        self,
        dike_section: DikeSection,
        revetment_data: RevetmentDataClass,
        target_beta: float,
        transition_level: float,
        evaluation_year: int,
    ) -> list[RevetmentMeasureData]:
        _evaluated_measures = []
        for _slope_part in revetment_data.slope_parts:
            _slope_relation = next(
                (
                    sr
                    for sr in _slope_part.slope_part_relations
                    if sr.year == evaluation_year
                ),
                None,
            )
            if _slope_part.end_part <= transition_level:
                _evaluated_measures.append(
                    self._get_stone_revetment_measure_data(
                        _slope_part, _slope_relation, target_beta, evaluation_year
                    )
                )
            elif (
                _slope_part.begin_part < transition_level
                and _slope_part.end_part > transition_level
            ):
                # TODO: this is not correct.
                _evaluated_measures.extend(
                    list(
                        self._get_combined_revetment_data(
                            _slope_part,
                            revetment_data,
                            _slope_relation,
                            transition_level,
                            evaluation_year,
                            target_beta,
                        )
                    )
                )
            elif _slope_part.begin_part >= transition_level:
                _evaluated_measures.append(
                    self._get_grass_revetment_data(
                        _slope_part, revetment_data, evaluation_year
                    )
                )
            else:
                raise ValueError(
                    "Can't evaluate revetment measure. Transition level: {}, begin part: {}, end_part: {}".format(
                        transition_level, _slope_part.begin_part, _slope_part.end_part
                    )
                )

        if transition_level >= max(
            map(lambda x: x.end_part, revetment_data.slope_parts)
        ):
            if transition_level >= dike_section.crest_height:
                raise ValueError("Overgang >= crest height")
            _extra_measure = RevetmentMeasureData(
                begin_part=transition_level,
                end_part=dike_section.crest_height,
                top_layer_type=20.0,
                previous_top_layer_type=float("nan"),
                top_layer_thickness=float("nan"),
                beta_block_revetment=float("nan"),
                beta_grass_revetment=self._evaluate_grass_revetment_data(
                    evaluation_year, revetment_data
                ),
                reinforce=True,
                tan_alpha=revetment_data.slope_parts[-1].end_part,
            )
            _evaluated_measures.append(_extra_measure)

        if transition_level > revetment_data.current_transition_level:
            self._correct_revetment_measure_data(_evaluated_measures, transition_level)

        return _evaluated_measures

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

    def _get_beta_target_vector(self, min_beta: float, p_max: float):
        _max_beta = pf_to_beta(p_max / self.max_pf_factor_block)
        return list(np.arange(min_beta, _max_beta, self.n_steps_block))

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
        _results_collection = RevetmentMeasureResultCollection()
        _results_collection.measure_id = self.parameters["ID"]
        _results_collection.measure_name = self.parameters["Name"]
        _results_collection.reinforcement_type = self.parameters["Type"]
        _results_collection.combinable_type = self.parameters["Class"]
        _results_collection.revetment_measure_results = self._get_intermediate_measures(
            dike_section, _revetment, _beta_targets, _transition_levels, self.config.T
        )

        # 4. Interpolate with years to calculate.

        # 5. Return Beta and Cost matrices
        self.measures = _results_collection

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

                # TODO: Should we include measures per year? Or just interpolate base on the vrtool.config years?
                _intermediate_measures.extend(
                    self._get_interpolated_measures(
                        _measures_per_year,
                        config_years,
                        revetment_years,
                    )
                )

        return _intermediate_measures

    def _get_measure_per_year(
        self,
        dike_section: DikeSection,
        revetment: RevetmentDataClass,
        beta_target: float,
        transition_level: float,
        measure_year: int,
    ):
        # 3.1. Get measure Beta and cost per year.
        _revetment_measures_collection = self._get_revetment_measure_data_collection(
            dike_section,
            revetment,
            beta_target,
            transition_level,
            measure_year,
        )
        _stone_beta_list, _grass_beta_list = zip(
            *(
                (rm.beta_block_revetment, rm.beta_grass_revetment)
                for rm in _revetment_measures_collection
            )
        )
        _combined_beta = RevetmentCalculator.calculate_combined_beta(
            _stone_beta_list, _grass_beta_list[0]
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

    def _evaluate_grass_revetment_data(
        self, evaluation_year: int, revetment: RevetmentDataClass
    ) -> float:
        return RevetmentCalculator.evaluate_grass_relations(
            evaluation_year,
            revetment.grass_relations,
            revetment.current_transition_level,
        )

    def _evaluate_stone_revetment_data(
        self, slope_part: SlopePartProtocol, evaluation_year: int
    ) -> float:
        return RevetmentCalculator.evaluate_block_relations(
            evaluation_year,
            slope_part.slope_part_relations,
            slope_part.top_layer_thickness,
        )
