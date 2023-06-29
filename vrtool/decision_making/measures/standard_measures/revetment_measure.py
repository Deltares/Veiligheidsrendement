import logging
import math
import numpy as np
from scipy import interpolate
from vrtool.common.dike_traject_info import DikeTrajectInfo
from vrtool.decision_making.measures.measure_protocol import MeasureProtocol
from vrtool.failure_mechanisms.revetment.relation_stone_revetment import (
    RelationStoneRevetment,
)
from vrtool.failure_mechanisms.revetment.revetment_calculator import (
    RevetmentCalculator,
)
from vrtool.failure_mechanisms.revetment.revetment_data_class import RevetmentDataClass
from vrtool.failure_mechanisms.revetment.slope_part import SlopePartProtocol
from vrtool.failure_mechanisms.revetment.slope_part.grass_slope_part import GRASS_TYPE
from vrtool.failure_mechanisms.revetment.slope_part.stone_slope_part import (
    StoneSlopePart,
)
from vrtool.flood_defence_system.dike_section import DikeSection


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


class RevetmentMeasureData:
    begin_part: float
    end_part: float
    top_layer_type: float
    previous_top_layer_type: float
    top_layer_thickness: float
    beta_block_revetment: list[float]
    beta_grass_revetment: list[float]
    reinforce: bool
    tan_alpha: float


class RevetmentMeasure(MeasureProtocol):

    transition_level_increase_step: float
    max_pf_factor_block: float
    n_steps_block: int

    def __init__(self, revetment_calculation: RevetmentCalculator) -> None:
        self.revetment_mechanism_calculator = revetment_calculation

    def _calculate_section_reliability(self) -> float:
        pass

    def _get_calculated_beta(self):
        return self.max_pf_factor_block * self.n_steps_block

    def _evaluate_grass_revetment_data(
        self, evaluation_year: int
    ) -> RevetmentMeasureData:
        return self.revetment_mechanism_calculator._evaluate_grass(evaluation_year)

    def _evaluate_stone_revetment_data(
        self, slope_part: SlopePartProtocol, evaluation_year: int
    ) -> RevetmentMeasureData:
        return self.revetment_mechanism_calculator._evaluate_block(
            slope_part, evaluation_year
        )

    def _design_steen(
        self,
        calculated_beta: float,
        top_layer_thickness: list[float],
        revetment_beta: list[float],
        top_layer_type: float,
        slope_top_thickness: float,
    ):
        _recalculated_beta = float("nan")
        _is_reinforced = False
        _top_layer_thickness = slope_top_thickness

        if not StoneSlopePart.is_stone_slope_part(top_layer_type):
            return _top_layer_thickness, _recalculated_beta, _is_reinforced

        # only for steen

        stone_interpolation = interpolate.interp1d(
            top_layer_thickness,
            np.array(revetment_beta) - calculated_beta,
            fill_value=("extrapolate"),
        )
        try:
            _top_layer_thickness = bisection(stone_interpolation, 0.0, 1.0, 0.01)
            _top_layer_thickness = math.ceil(_top_layer_thickness / 0.05) * 0.05
            _recalculated_beta = calculated_beta
            _is_reinforced = True
        except:
            _top_layer_thickness = float("nan")

        if _top_layer_thickness <= slope_top_thickness:
            logging.warn("Design D is <= than the current D")
            _top_layer_thickness = slope_top_thickness
            _recalculated_beta = self._evaluate_stone_revetment_data(
                # slope_part, evaluation_year # TODO, not clear how to set these values
            )

        return _top_layer_thickness, _recalculated_beta, _is_reinforced

    def get_stone_revetment_measure_data(
        self, slope_part: SlopePartProtocol, stone_revetment: RelationStoneRevetment
    ) -> RevetmentMeasureData:
        (
            _new_top_layer_thickness,
            _new_beta_block_revetment,
            _is_reinforced,
        ) = self._get_design_steen(
            self._get_calculated_beta(),
            stone_revetment.top_layer_thickness,
            stone_revetment.beta,
            slope_part.top_layer_type,
            slope_part.top_layer_thickness,
        )
        _revetment_measure_data = RevetmentMeasureData()

        _revetment_measure_data.begin_part = slope_part.begin_part
        _revetment_measure_data.end_part = slope_part.end_part
        _revetment_measure_data.top_layer_type = slope_part.top_layer_type
        _revetment_measure_data.previous_top_layer_type = slope_part.top_layer_type
        _revetment_measure_data.top_layer_thickness = _new_top_layer_thickness
        _revetment_measure_data.beta_block_revetment = _new_beta_block_revetment
        _revetment_measure_data.beta_grass_revetment = float("nan")
        _revetment_measure_data.reinforce = _is_reinforced
        _revetment_measure_data.tan_alpha = slope_part.tan_alpha
        return _revetment_measure_data

    def get_combined_revetment_data(
        self,
        slope_part: SlopePartProtocol,
        stone_revetment: RelationStoneRevetment,
        transition_level,
    ) -> tuple[RevetmentMeasureData, RevetmentMeasureData]:
        (
            _new_top_layer_thickness,
            _new_beta_block_revetment,
            _is_reinforced,
        ) = self._get_design_steen(
            self._get_calculated_beta(),
            stone_revetment.top_layer_thickness,
            stone_revetment.beta,
            slope_part.top_layer_type,
            slope_part.top_layer_thickness,
        )
        _stone_rmd = RevetmentMeasureData()
        _stone_rmd.begin_part = slope_part.begin_part
        _stone_rmd.end_part = transition_level
        _stone_rmd.top_layer_type = slope_part.top_layer_type
        _stone_rmd.previous_top_layer_type = slope_part.top_layer_type
        _stone_rmd.top_layer_thickness = _new_top_layer_thickness
        _stone_rmd.beta_block_revetment = _new_beta_block_revetment
        _stone_rmd.beta_grass_revetment = float("nan")
        _stone_rmd.reinforce = _is_reinforced
        _stone_rmd.tan_alpha = slope_part.tan_alpha

        _grass_rmd = RevetmentMeasureData()
        _grass_rmd.begin_part = transition_level
        _grass_rmd.end_part = slope_part.end_part
        _grass_rmd.top_layer_type = GRASS_TYPE
        _grass_rmd.previous_top_layer_type = slope_part.top_layer_type
        _grass_rmd.top_layer_thickness = float("nan")
        _grass_rmd.beta_block_revetment = float("nan")
        _grass_rmd.beta_grass_revetment = self._evaluate_grass_revetment_data(
            evaluation_year
        )
        _grass_rmd.reinforce = True
        _grass_rmd.tan_alpha = slope_part.tan_alpha

        return _stone_rmd, _grass_rmd

    def get_grass_revetment_data(
        self, slope_part: SlopePartProtocol
    ) -> RevetmentMeasureData:
        _grass_rmd = RevetmentMeasureData()
        _grass_rmd.begin_part = slope_part.begin_part
        _grass_rmd.end_part = slope_part.end_part
        _grass_rmd.top_layer_type = GRASS_TYPE
        _grass_rmd.previous_top_layer_type = slope_part.top_layer_type
        _grass_rmd.top_layer_thickness = float("nan")
        _grass_rmd.beta_block_revetment = float("nan")
        _grass_rmd.beta_grass_revetment = self._evaluate_grass_revetment_data(
            evaluation_year
        )
        _grass_rmd.reinforce = True
        _grass_rmd.tan_alpha = slope_part.tan_alpha
        return _grass_rmd

    def evaluate_revetment_measure(
        self,
        revetment_data_class: RevetmentDataClass,
        beta,
        transition_level,
        evaluation_year: int,
    ) -> list[RevetmentMeasureData]:
        _evaluated_measures = []
        for _slope_part in revetment_data_class.slope_parts:
            if _slope_part.end_part <= transition_level:
                _evaluated_measures.append(self.get_stone_revetment_measure_data(beta))
            elif (
                _slope_part.begin_part < transition_level
                and _slope_part.end_part > transition_level
            ):
                _evaluated_measures.extend(list(self.get_combined_revetment_data(beta)))
            elif _slope_part.begin_part >= transition_level:
                _evaluated_measures.append(self.get_grass_revetment_data(beta))
            # TODO: Check, raise an error or just log? Original script raises nothing.
            raise ValueError("Can't evaluate revetment measure.")

        if transition_level >= max(
            lambda x: x.end_part, revetment_data_class.slope_parts
        ):
            # TODO: Verify where `self.crest_height` parameter should come from.
            _crest_height = self.crest_height
            if transition_level >= _crest_height:
                raise ValueError("Overgang >= kruinhoogte")
            _extra_measure = RevetmentMeasureData()
            _extra_measure.begin_part = transition_level
            _extra_measure.end_part = _crest_height
            _extra_measure.top_layer_type = 20.0
            _extra_measure.previous_top_layer_type = float("nan")
            _extra_measure.top_layer_thickness = float("nan")
            _extra_measure.beta_block_revetment = float("nan")
            _extra_measure.beta_grass_revetment = self._evaluate_grass_revetment_data(
                evaluation_year
            )
            _extra_measure.reinforce = True
            _extra_measure.tan_alpha = revetment_data_class.slope_parts[-1].end_part

        return _evaluated_measures

    def evaluate_measure(
        self,
        dike_section: DikeSection,
        traject_info: DikeTrajectInfo,
        preserve_slope: bool,
    ):
        self.measures = {}
        self.measures["Reliability"] = self._calculate_section_reliability()
        self.measures["Cost"] = self.revetment_mechanism_calculator.calculate_cost(
            dike_section.Length, year
        )

    def _evaluate_stone_slope(
        self, slope_part: SlopePartProtocol, d_opt, beta, beta_failure
    ):

        fsteen = interpolate.interp1d(
            d_opt, np.array(beta_failure) - beta, fill_value=("extrapolate")
        )
        try:
            topd = bisection(fsteen, 0.0, 1.0, 0.01)
            topd = math.ceil(topd / 0.05) * 0.05
            betares = beta
            reinforce = "yes"
        except:
            topd = np.nan
            betares = np.nan
            reinforce = "no"

        if topd <= D:
            warnings.warn("Design D is <= than the current D")
            topd = D
            betares = evaluate_steen(topd, D_opt, betaFalen)
            reinforce = "no"
        measure["Zo"].append(dataZST["Zo"][i])
        measure["Zb"].append(dataZST["Zb"][i])
        measure["toplaagtype"].append(dataZST["toplaagtype"][i])
        measure["previous toplaagtype"].append(dataZST["toplaagtype"][i])
        measure["D"].append(topd)
        measure["betaZST"].append(betares)
        measure["betaGEBU"].append(np.nan)
        measure["reinforce"].append(reinforce)
        measure["tana"].append(dataZST["tana"][i])

    def _evaluate_measure_karolina_script(self):
        measure = {
            "Zo": list(),
            "Zb": list(),
            "toplaagtype": list(),
            "D": list(),
            "betaZST": list(),
            "betaGEBU": list(),
            "previous toplaagtype": list(),
            "reinforce": list(),
            "tana": list(),
        }

        count = 0
        for i in range(0, dataZST["aantal deelvakken"]):

            if dataZST["Zb"][i] <= h_onder:

                count += 1
                # steen
                topd, betares, reinforce = design_steen(
                    beta,
                    dataZST[f"deelvak {i}"]["D_opt"],
                    dataZST[f"deelvak {i}"]["betaFalen"],
                    dataZST["toplaagtype"][i],
                    dataZST["D huidig"][i],
                )

                measure["Zo"].append(dataZST["Zo"][i])
                measure["Zb"].append(dataZST["Zb"][i])
                measure["toplaagtype"].append(dataZST["toplaagtype"][i])
                measure["previous toplaagtype"].append(dataZST["toplaagtype"][i])
                measure["D"].append(topd)
                measure["betaZST"].append(betares)
                measure["betaGEBU"].append(np.nan)
                measure["reinforce"].append(reinforce)
                measure["tana"].append(dataZST["tana"][i])

            elif (
                dataZST["Zo"][i] < h_onder and dataZST["Zb"][i] > h_onder
            ):  # part is steen and part is gras

                count += 2
                # steen
                topd, betares, reinforce = design_steen(
                    beta,
                    dataZST[f"deelvak {i}"]["D_opt"],
                    dataZST[f"deelvak {i}"]["betaFalen"],
                    dataZST["toplaagtype"][i],
                    dataZST["D huidig"][i],
                )

                measure["Zo"].append(dataZST["Zo"][i])
                measure["Zb"].append(h_onder)
                measure["toplaagtype"].append(dataZST["toplaagtype"][i])
                measure["previous toplaagtype"].append(dataZST["toplaagtype"][i])
                measure["D"].append(topd)
                measure["betaZST"].append(betares)
                measure["betaGEBU"].append(np.nan)
                measure["reinforce"].append(reinforce)
                measure["tana"].append(dataZST["tana"][i])

                # gras
                measure["Zo"].append(h_onder)
                measure["Zb"].append(dataZST["Zb"][i])
                measure["toplaagtype"].append(20.0)
                measure["previous toplaagtype"].append(dataZST["toplaagtype"][i])
                measure["D"].append(np.nan)
                measure["betaZST"].append(np.nan)
                measure["betaGEBU"].append(
                    evaluate_gras(
                        h_onder, dataGEBU["grasbekleding_begin"], dataGEBU["betaFalen"]
                    )
                )
                measure["reinforce"].append("yes")
                measure["tana"].append(dataZST["tana"][i])

            elif dataZST["Zo"][i] >= h_onder:  # gras

                count += 1
                # gras
                measure["Zo"].append(dataZST["Zo"][i])
                measure["Zb"].append(dataZST["Zb"][i])
                measure["toplaagtype"].append(20.0)
                measure["previous toplaagtype"].append(dataZST["toplaagtype"][i])
                measure["D"].append(np.nan)
                measure["betaZST"].append(np.nan)
                measure["betaGEBU"].append(
                    evaluate_gras(
                        h_onder, dataGEBU["grasbekleding_begin"], dataGEBU["betaFalen"]
                    )
                )
                measure["reinforce"].append("yes")
                measure["tana"].append(dataZST["tana"][i])

        if h_onder >= np.max(dataZST["Zb"]):

            if h_onder >= kruinhoogte:
                raise ValueError("Overgang >= kruinhoogte")

            count += 1
            # gras
            measure["Zo"].append(h_onder)
            measure["Zb"].append(kruinhoogte)
            measure["toplaagtype"].append(20.0)
            measure["previous toplaagtype"].append(np.nan)
            measure["D"].append(np.nan)
            measure["betaZST"].append(np.nan)
            measure["betaGEBU"].append(
                evaluate_gras(
                    h_onder, dataGEBU["grasbekleding_begin"], dataGEBU["betaFalen"]
                )
            )
            measure["reinforce"].append("yes")
            measure["tana"].append(dataZST["tana"][i])

        return count, measure
