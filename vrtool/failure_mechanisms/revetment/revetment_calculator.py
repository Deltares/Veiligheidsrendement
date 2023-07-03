import numpy as np
from scipy.interpolate import interp1d
from scipy.special import ndtri

from vrtool.failure_mechanisms.failure_mechanism_calculator_protocol import (
    FailureMechanismCalculatorProtocol,
)
from vrtool.failure_mechanisms.revetment.revetment_data_class import RevetmentDataClass
from vrtool.failure_mechanisms.revetment.slope_part import (
    GrassSlopePart,
    StoneSlopePart,
)
from vrtool.probabilistic_tools.probabilistic_functions import beta_to_pf


class RevetmentCalculator(FailureMechanismCalculatorProtocol):
    def __init__(self, revetment: RevetmentDataClass) -> None:
        self._revetment = revetment

    def calculate(self, year: int) -> tuple[float, float]:
        _given_years = self._revetment.find_given_years()
        _beta_per_year = []
        for given_year in _given_years:
            _stone_revetment_beta = []
            _grass_revetment_beta = np.nan
            for _slope_part in self._revetment.slope_parts:
                if isinstance(_slope_part, StoneSlopePart):
                    _stone_revetment_beta.append(
                        self._evaluate_block(_slope_part, given_year)
                    )
                elif isinstance(_slope_part, GrassSlopePart) and np.isnan(
                    _grass_revetment_beta
                ):
                    _stone_revetment_beta.append(np.nan)
                    _grass_revetment_beta = self._evaluate_grass(given_year)
                else:
                    _stone_revetment_beta.append(np.nan)
            _beta_per_year.append(
                self._calculate_combined_beta(
                    _stone_revetment_beta, _grass_revetment_beta
                )
            )

        if len(_given_years) == 1:
            return _beta_per_year[0], beta_to_pf(_beta_per_year[0])

        _interpolate_beta = interp1d(
            _given_years, _beta_per_year, fill_value=("extrapolate")
        )
        _calculated_beta = _interpolate_beta(year)
        return _calculated_beta, beta_to_pf(_calculated_beta)

    def _calculate_combined_beta(
        self, stone_revetment_beta: list[float], grass_revetment_beta: float
    ) -> float:
        if np.all(np.isnan(stone_revetment_beta)):
            _prob_stone_revetment = 0.0
        else:
            _prob_stone_revetment = beta_to_pf(np.nanmin(stone_revetment_beta))

        if np.isnan(grass_revetment_beta):
            _prob_grass_revetment = 0.0
        else:
            _prob_grass_revetment = beta_to_pf(grass_revetment_beta)

        _prob_combined = _prob_stone_revetment + _prob_grass_revetment
        _beta_combined = -ndtri(_prob_combined)
        return _beta_combined

    def _evaluate_block(self, slope_part: StoneSlopePart, given_year: int):
        D_opt = []
        beta_failure = []
        for _slope_part_relation in slope_part.slope_part_relations:
            if _slope_part_relation.year == given_year:
                D_opt.append(_slope_part_relation.top_layer_thickness)
                beta_failure.append(_slope_part_relation.beta)

        fBlock = interp1d(D_opt, beta_failure, fill_value=("extrapolate"))
        beta = fBlock(slope_part.top_layer_thickness)

        return beta

    def _evaluate_grass(self, given_year: int):
        transitions = []
        betaFailure = []
        for rel in self._revetment.grass_relations:
            if rel.year == given_year:
                transitions.append(rel.transition_level)
                betaFailure.append(rel.beta)

        fgrass = interp1d(transitions, betaFailure, fill_value=("extrapolate"))
        beta = fgrass(self._revetment.current_transition_level)

        return beta
