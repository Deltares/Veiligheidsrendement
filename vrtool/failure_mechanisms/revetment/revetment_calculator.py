import numpy as np
from scipy.interpolate import interp1d
from scipy.special import ndtri

from vrtool.failure_mechanisms.failure_mechanism_calculator_protocol import (
    FailureMechanismCalculatorProtocol,
)
from vrtool.failure_mechanisms.revetment.relation_grass_revetment import (
    RelationGrassRevetment,
)
from vrtool.failure_mechanisms.revetment.relation_stone_revetment import (
    RelationStoneRevetment,
)
from vrtool.failure_mechanisms.revetment.revetment_data_class import RevetmentDataClass
from vrtool.failure_mechanisms.revetment.slope_part import (
    GrassSlopePart,
    StoneSlopePart,
)
from vrtool.probabilistic_tools.probabilistic_functions import beta_to_pf


class RevetmentCalculator(FailureMechanismCalculatorProtocol):
    def __init__(self, revetment: RevetmentDataClass, initial_year: int) -> None:
        self._revetment = revetment
        self._initial_year = initial_year

    def calculate(self, year: int) -> tuple[float, float]:
        _given_years = self._revetment.get_available_years()
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
        _calculated_beta = _interpolate_beta(self._initial_year + year)
        return _calculated_beta, beta_to_pf(_calculated_beta)

    def _calculate_combined_beta(
        self, stone_revetment_beta: list[float], grass_revetment_beta: float
    ) -> float:
        return self.calculate_combined_beta(stone_revetment_beta, grass_revetment_beta)

    def _evaluate_block(self, slope_part: StoneSlopePart, given_year: int):
        return self.evaluate_block_relations(
            given_year, slope_part.slope_part_relations, slope_part.top_layer_thickness
        )

    def _evaluate_grass(self, given_year: int):
        return self.evaluate_grass_relations(
            given_year,
            self._revetment.grass_relations,
            self._revetment.current_transition_level,
        )

    @staticmethod
    def calculate_combined_beta(
        stone_revetment_beta: list[float], grass_revetment_beta: float
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

    @staticmethod
    def evaluate_grass_relations(
        evaluation_year: int,
        grass_relations: list[RelationGrassRevetment],
        current_transition_level: float,
    ) -> float:
        _transitions, _beta_failure = zip(
            *(
                (grass_relation.transition_level, grass_relation.beta)
                for grass_relation in grass_relations
                if grass_relation.year == evaluation_year
            )
        )

        _interpolate_grass = interp1d(
            _transitions, _beta_failure, fill_value=("extrapolate")
        )
        return _interpolate_grass(current_transition_level)

    @staticmethod
    def evaluate_block_relations(
        evaluation_year: int,
        slope_part_relations: list[RelationStoneRevetment],
        top_layer_thickness: float,
    ) -> float:
        _top_layer_thickness, _beta_failure = zip(
            *(
                (slope_relation.top_layer_thickness, slope_relation.beta)
                for slope_relation in slope_part_relations
                if slope_relation.year == evaluation_year
            )
        )

        _interpolate_block = interp1d(
            _top_layer_thickness, _beta_failure, fill_value=("extrapolate")
        )
        return _interpolate_block(top_layer_thickness)
