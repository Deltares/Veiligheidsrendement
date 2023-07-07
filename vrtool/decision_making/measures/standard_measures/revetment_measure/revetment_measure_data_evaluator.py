from vrtool.decision_making.measures.standard_measures.revetment_measure.revetment_measure_data import (
    RevetmentMeasureData,
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
import math

from math import isnan
import numpy as np
from scipy.interpolate import interp1d
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


class RevetmentMeasureDataBuilder:
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
            _top_layer_thickness = slope_part.top_layer_thickness
            _recalculated_beta = float(
                self.evaluate_stone_revetment_data(slope_part, evaluation_year)
            )
            _is_reinforced = False

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
            beta_grass_revetment=self.evaluate_grass_revetment_data(
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
            beta_grass_revetment=self.evaluate_grass_revetment_data(
                evaluation_year, revetment_data
            ),
            reinforce=True,
            tan_alpha=slope_part.tan_alpha,
        )

    def build_revetment_measure_data_collection(
        self,
        crest_height: float,
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
            if transition_level >= crest_height:
                raise ValueError("Overgang >= crest height")
            _extra_measure = RevetmentMeasureData(
                begin_part=transition_level,
                end_part=crest_height,
                top_layer_type=20.0,
                previous_top_layer_type=float("nan"),
                top_layer_thickness=float("nan"),
                beta_block_revetment=float("nan"),
                beta_grass_revetment=self.evaluate_grass_revetment_data(
                    evaluation_year, revetment_data
                ),
                reinforce=True,
                tan_alpha=revetment_data.slope_parts[-1].end_part,
            )
            _evaluated_measures.append(_extra_measure)

        if transition_level > revetment_data.current_transition_level:
            self._correct_revetment_measure_data(_evaluated_measures, transition_level)

        return _evaluated_measures

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
