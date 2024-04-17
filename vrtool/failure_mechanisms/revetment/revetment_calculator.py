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
                        self.evaluate_block_relations(
                            given_year,
                            _slope_part.slope_part_relations,
                            _slope_part.top_layer_thickness,
                        )
                    )
                elif isinstance(_slope_part, GrassSlopePart) and np.isnan(
                    _grass_revetment_beta
                ):
                    _stone_revetment_beta.append(np.nan)
                    _grass_revetment_beta = self.evaluate_grass_relations(
                        given_year,
                        self._revetment.grass_relations,
                        self._revetment.current_transition_level,
                    )
                else:
                    _stone_revetment_beta.append(np.nan)
            _beta_per_year.append(
                self.calculate_combined_beta(
                    _stone_revetment_beta, _grass_revetment_beta
                )
            )

        self._revetment.set_beta_stone(np.nanmin(_stone_revetment_beta))

        if len(_given_years) == 1:
            return _beta_per_year[0], beta_to_pf(_beta_per_year[0])

        _interpolate_beta = interp1d(
            _given_years, _beta_per_year, fill_value=("extrapolate")
        )
        _calculated_beta = _interpolate_beta(self._initial_year + year)
        return _calculated_beta, beta_to_pf(_calculated_beta)

    @staticmethod
    def calculate_combined_beta(
        stone_revetment_beta: list[float], grass_revetment_beta: float
    ) -> float:
        """
        Calculates the combined beta for all the available Stone Revetments and a related Grass Revetment.

        Args:
            stone_revetment_beta (list[float]): List of Stone Revetment beta values.
            grass_revetment_beta (float): Grass revetment beta value.

        Returns:
            float: Combined beta values.
        """
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
        """
        Evaluates all given grass relations for a given `evaluation_year` and a concrete `current_transition_level`.

        Args:
            evaluation_year (int): Evaluation year to apply to the `grass_relations`.
            grass_relations (list[RelationGrassRevetment]): List of grass revetment relations.
            current_transition_level (float): transition level required to interpolate the grass relations.

        Returns:
            float: Interpolated betas for the grass relations, evaluation year and current transition level.
        """
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
        """
        Evaluates all given slope part relations for a given `evaluation_year` and a concrete `top_layer_thickness`.

        Args:
            evaluation_year (int): Evaluation year to apply to the `slope_part_relations`.
            slope_part_relations (list[RelationStoneRevetment]): List of slope part relations.
            top_layer_thickness (float): Thickness value for the top most layer to interpolate the slope part relations.

        Returns:
            float:  Interpolated betas for the slope part relations, evaluation year and top layer thickness.
        """
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
