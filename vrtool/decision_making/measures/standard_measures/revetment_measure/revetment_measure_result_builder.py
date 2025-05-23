from math import isnan

import numpy as np
from scipy.interpolate import interp1d

from vrtool.common.measure_unit_costs import MeasureUnitCosts
from vrtool.decision_making.measures.standard_measures.revetment_measure.revetment_measure_data import (
    RevetmentMeasureData,
)
from vrtool.decision_making.measures.standard_measures.revetment_measure.revetment_measure_result import (
    RevetmentMeasureResult,
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


class RevetmentMeasureResultBuilder:
    def build(
        self,
        crest_height: float,
        dike_length: float,
        revetment: RevetmentDataClass,
        beta_target: float,
        transition_level: float,
        measure_year: int,
        unit_costs: MeasureUnitCosts,
    ) -> RevetmentMeasureResult:
        """
        Creates a valid instance of a `RevetmentMeasureResult` based on all the given arguments.

        Args:
            crest_height (float): Dike section crest height.
            dike_length (float): Dike length.
            revetment (RevetmentDataClass): Revetment data describing all properties for all the possible slope and grass parts (`SlopePartProtocol` and `RelationGrassRevetment`).
            beta_target (float): Desired beta set as a limit.
            transition_level (float): Current transition level.
            measure_year (int): Year for which calculation is being done.

        Returns:
            RevetmentMeasureResult: Simple dataclass containing the result data of a (re)calculated revetment measure.
        """
        # 3.1. Get measure Beta and cost per year.
        _revetment_measures_collection = self._get_revetment_measures_collection(
            crest_height,
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
        # Get the simple grass beta.
        _grass_beta = self._get_grass_revetment_beta_from_vector(_grass_beta_list)
        _combined_beta = RevetmentCalculator.calculate_combined_beta(
            _stone_beta_list, _grass_beta
        )
        _cost = sum(
            map(
                lambda x: x.get_total_cost(dike_length, unit_costs),
                _revetment_measures_collection,
            )
        )
        return RevetmentMeasureResult(
            year=measure_year,
            beta_target=beta_target,
            beta_combined=_combined_beta,
            transition_level=transition_level,
            cost=_cost,
        )

    def _get_revetment_measures_collection(
        self,
        crest_height: float,
        revetment_data: RevetmentDataClass,
        target_beta: float,
        transition_level: float,
        evaluation_year: int,
    ) -> list[RevetmentMeasureData]:
        self._transition_level = transition_level
        _evaluated_measures = []
        if not revetment_data or not any(revetment_data.slope_parts):
            return _evaluated_measures

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

        _max_end_part = max(map(lambda x: x.end_part, revetment_data.slope_parts))
        if transition_level >= _max_end_part:
            if transition_level >= crest_height:
                raise ValueError("Overgang >= crest height")
            # Unknown
            _extra_unknown_measure = RevetmentMeasureData(
                begin_part=_max_end_part,
                end_part=transition_level,
                top_layer_type=float("nan"),
                previous_top_layer_type=20.0,
                top_layer_thickness=float("nan"),
                beta_block_revetment=float("nan"),
                beta_grass_revetment=float("nan"),
                reinforce=True,
                tan_alpha=revetment_data.slope_parts[-1].tan_alpha,
            )
            _evaluated_measures.append(_extra_unknown_measure)

            # Grass
            _extra_grass_measure = RevetmentMeasureData(
                begin_part=transition_level,
                end_part=crest_height,
                top_layer_type=20.0,
                previous_top_layer_type=float("nan"),
                top_layer_thickness=float("nan"),
                beta_block_revetment=float("nan"),
                beta_grass_revetment=self._evaluate_grass_revetment_data(
                    evaluation_year, revetment_data
                ),
                reinforce=True,
                tan_alpha=revetment_data.slope_parts[-1].tan_alpha,
            )
            _evaluated_measures.append(_extra_grass_measure)

        if transition_level > revetment_data.current_transition_level:
            self._correct_revetment_measure_data(_evaluated_measures, transition_level)

        return _evaluated_measures

    def _correct_revetment_measure_data(
        self,
        revetment_measures: list[RevetmentMeasureData],
        current_transition_level: float,
    ) -> None:

        _stone_revetments = [
            rm
            for rm in revetment_measures
            if StoneSlopePart.is_stone_slope_part(rm.top_layer_type)
        ]
        if not _stone_revetments:
            raise ValueError("No stone revetment measure was found.")

        _last_stone_revetment = _stone_revetments[-1]

        for _revetment_measure in revetment_measures:
            if (
                GrassSlopePart.is_grass_part(_revetment_measure.previous_top_layer_type)
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
            _top_layer_thickness = bisection(stone_interpolation, 0.0, 1.0, 0.001)
            _recalculated_beta = calculated_beta
            _is_reinforced = True
        except:
            _top_layer_thickness = slope_part.top_layer_thickness

        if (
            _top_layer_thickness <= slope_part.top_layer_thickness
            or np.abs(_top_layer_thickness - slope_part.top_layer_thickness) <= 0.01
        ):
            # We compare whether the difference is less than 0.01 meter (1cm).
            _top_layer_thickness = slope_part.top_layer_thickness
            _recalculated_beta = float(
                self._evaluate_stone_revetment_data(slope_part, evaluation_year)
            )
            _is_reinforced = False

        if _top_layer_thickness <= 0.0:
            raise ValueError("Negative top layer thickness found.")

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

    def _evaluate_grass_revetment_data(
        self, evaluation_year: int, revetment: RevetmentDataClass
    ) -> float:
        return float(
            RevetmentCalculator.evaluate_grass_relations(
                evaluation_year,
                revetment.grass_relations,
                self._transition_level,
            )
        )

    def _evaluate_stone_revetment_data(
        self, slope_part: SlopePartProtocol, evaluation_year: int
    ) -> float:
        return RevetmentCalculator.evaluate_block_relations(
            evaluation_year,
            slope_part.slope_part_relations,
            slope_part.top_layer_thickness,
        )

    def _get_grass_revetment_beta_from_vector(
        self, grass_revetment_betas: list[float]
    ) -> float:
        # The `grass_revetment_betas` contain a list where only the last values are valid.
        # At the same time, these last values are always the same.
        return float(
            next(
                (
                    grass_beta
                    for grass_beta in grass_revetment_betas
                    if not isnan(grass_beta)
                ),
                "nan",
            )
        )
