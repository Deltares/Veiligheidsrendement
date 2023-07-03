import copy
import logging
import math

import numpy as np
from scipy.interpolate import interp1d
from scipy.special import ndtri
from scipy.stats import norm

from vrtool.common.dike_traject_info import DikeTrajectInfo
from vrtool.decision_making.measures.measure_protocol import MeasureProtocol
from vrtool.failure_mechanisms.revetment.relation_stone_revetment import (
    RelationStoneRevetment,
)
from vrtool.failure_mechanisms.revetment.revetment_calculator import RevetmentCalculator
from vrtool.failure_mechanisms.revetment.revetment_data_class import RevetmentDataClass
from vrtool.failure_mechanisms.revetment.slope_part import SlopePartProtocol
from vrtool.failure_mechanisms.revetment.slope_part.grass_slope_part import (
    GRASS_TYPE,
    GrassSlopePart,
)
from vrtool.failure_mechanisms.revetment.slope_part.stone_slope_part import (
    StoneSlopePart,
)
from vrtool.flood_defence_system.dike_section import DikeSection
from vrtool.flood_defence_system.mechanism_reliability import MechanismReliability
from vrtool.flood_defence_system.mechanism_reliability_collection import (
    MechanismReliabilityCollection,
)
from vrtool.flood_defence_system.section_reliability import SectionReliability


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


class RevetmentReliability:
    measure_data: list[RevetmentMeasureData]
    reliability: MechanismReliability
    cost: float


class RevetmentMeasure(MeasureProtocol):
    revetment: RevetmentDataClass
    _revetment_reliability_collection: list[RevetmentReliability]

    def __init__(self) -> None:
        self._revetment_reliability_collection = []

    @property
    def transition_level_increase_step(self) -> float:
        return self.parameters["transition_level_increase_step"]

    @property
    def max_pf_factor_block(self) -> float:
        return self.parameters["max_pf_factor_block"]

    @property
    def n_steps_block(self) -> int:
        return self.parameters["n_steps_block"]

    def _get_configured_section_reliability(
        self,
        dike_section: DikeSection,
        traject_info: DikeTrajectInfo,
    ) -> SectionReliability:
        # TODO: Same as in stability_screen_measure.
        # Technically there we are already retrieving these mechanism_reliability_collections.
        # We should either generate them once or leave the creation responsibility to each measure.

        section_reliability = SectionReliability()
        mechanism_names = (
            dike_section.section_reliability.failure_mechanisms.get_available_mechanisms()
        )
        _revetment_mechanism_name = "revetment"
        if _revetment_mechanism_name not in mechanism_names:
            # TODO: Technically there's only one mechanism possible for this.
            return section_reliability

        calc_type = dike_section.mechanism_data[_revetment_mechanism_name][0][1]
        mechanism_reliability_collection = (
            self._get_configured_mechanism_reliability_collection(
                _revetment_mechanism_name,
                calc_type,
                dike_section,
                traject_info,
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
    ) -> MechanismReliabilityCollection:
        # TODO: Almost duplicated code from other measures.
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

            # This call calculates the Reliability of the dike section.
            mechanism_reliability_collection.generate_LCR_profile(
                dike_section.section_reliability.load,
                traject_info=traject_info,
            )

            mechanism_reliability = mechanism_reliability_collection.Reliability[
                year_to_calculate
            ]
            self._configure_revetment(
                mechanism_reliability, year_to_calculate, dike_section
            )

        return mechanism_reliability_collection

    def _configure_revetment(
        self,
        mechanism_reliability: MechanismReliability,
        year_to_calculate: str,
        dike_section: DikeSection,
    ) -> None:

        _calculated_beta = self._get_calculated_beta()
        _revetment_measures_collection = self._get_revetment_measure_data_collection(
            dike_section,
            self.revetment,
            _calculated_beta,
            self.transition_level_increase_step,
            int(year_to_calculate),
        )
        # TODO this is meant to be set through the calculator.
        # _stone_beta_list, _grass_beta_list = zip(
        #     *(
        #         (rm.beta_block_revetment, rm.beta_grass_revetment)
        #         for rm in _revetment_measures_collection
        #     )
        # )
        # mechanism_reliability["BETA"] = self._calculate_combined_beta(
        #     _stone_beta_list, _grass_beta_list
        # )
        # This should be simplified so we call directly the revetment calculation.
        mechanism_reliability.calculate_reliability()
        _revetment_reliability = RevetmentReliability()
        _revetment_reliability.measure_data = _revetment_measures_collection
        _revetment_reliability.reliability = mechanism_reliability
        _revetment_reliability.cost = self._calculate_year_costs(
            _revetment_measures_collection, dike_section.Length, int(year_to_calculate)
        )
        self._revetment_reliability_collection.append(_revetment_reliability)

    def _get_calculated_beta(self):
        return self.max_pf_factor_block * self.n_steps_block

    def _correct_revetment_measure_data(
        self,
        revetment_measures: list[RevetmentMeasureData],
        current_transition_level: float,
    ) -> RevetmentMeasureData:
        _last_stone_revetment = next(
            (
                rm
                for rm in revetment_measures
                if StoneSlopePart.is_stone_slope_part(rm.top_layer_type)
            ),
            None,
        )
        if not _last_stone_revetment:
            raise ValueError("No stone revetment measure was found.")

        for _grass_revetment in revetment_measures:
            if (
                GrassSlopePart.is_grass_part(_grass_revetment.top_layer_type)
                and _grass_revetment.begin_part < current_transition_level
            ):
                _grass_revetment.top_layer_thickness = (
                    _last_stone_revetment.top_layer_thickness
                )
                _grass_revetment.top_layer_type = 2026.0
                _grass_revetment.beta_block_revetment = (
                    _last_stone_revetment.beta_block_revetment
                )
                _grass_revetment.reinforce = True

    def _get_design_stone(
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

        stone_interpolation = interp1d(
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

    def _get_stone_revetment_measure_data(
        self, slope_part: SlopePartProtocol, stone_revetment: RelationStoneRevetment
    ) -> RevetmentMeasureData:
        (
            _new_top_layer_thickness,
            _new_beta_block_revetment,
            _is_reinforced,
        ) = self._get_design_stone(
            self._get_calculated_beta(),
            stone_revetment.top_layer_thickness,
            stone_revetment.beta,
            slope_part.top_layer_type,
            slope_part.top_layer_thickness,
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
        stone_revetment: RelationStoneRevetment,
        transition_level,
        evaluation_year: int,
    ) -> tuple[RevetmentMeasureData, RevetmentMeasureData]:
        (
            _new_top_layer_thickness,
            _new_beta_block_revetment,
            _is_reinforced,
        ) = self._get_design_stone(
            self._get_calculated_beta(),
            stone_revetment.top_layer_thickness,
            stone_revetment.beta,
            slope_part.top_layer_type,
            slope_part.top_layer_thickness,
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
            beta_grass_revetment=self._evaluate_grass_revetment_data(evaluation_year),
            reinforce=True,
            tan_alpha=slope_part.tan_alpha,
        )

        return _stone_rmd, _grass_rmd

    def _get_grass_revetment_data(
        self, slope_part: SlopePartProtocol, evaluation_year: int
    ) -> RevetmentMeasureData:
        return RevetmentMeasureData(
            begin_part=slope_part.begin_part,
            end_part=slope_part.end_part,
            top_layer_type=GRASS_TYPE,
            previous_top_layer_type=slope_part.top_layer_type,
            top_layer_thickness=float("nan"),
            beta_block_revetment=float("nan"),
            beta_grass_revetment=self._evaluate_grass_revetment_data(evaluation_year),
            reinforce=True,
            tan_alpha=slope_part.tan_alpha,
        )

    def _get_revetment_measure_data_collection(
        self,
        dike_section: DikeSection,
        revetment_data_class: RevetmentDataClass,
        beta,
        transition_level: float,
        evaluation_year: int,
    ) -> list[RevetmentMeasureData]:
        _evaluated_measures = []
        for _slope_part in revetment_data_class.slope_parts:
            if _slope_part.end_part <= transition_level:
                _evaluated_measures.append(
                    self._get_stone_revetment_measure_data(
                        _slope_part, _slope_part.slope_part_relations
                    )
                )
            elif (
                _slope_part.begin_part < transition_level
                and _slope_part.end_part > transition_level
            ):
                _evaluated_measures.extend(
                    list(self._get_combined_revetment_data(beta))
                )
            elif _slope_part.begin_part >= transition_level:
                _evaluated_measures.append(
                    self._get_grass_revetment_data(_slope_part, evaluation_year)
                )
            else:
                raise ValueError(
                    "Can't evaluate revetment measure. Transition level: {}, begin part: {}, end_part: {}".format(
                        transition_level, _slope_part.begin_part, _slope_part.end_part
                    )
                )

        if transition_level >= max(
            lambda x: x.end_part, revetment_data_class.slope_parts
        ):
            if transition_level >= dike_section.crest_height:
                raise ValueError("Overgang >= kruinhoogte")
            _extra_measure = RevetmentMeasureData(
                begin_part=transition_level,
                end_part=dike_section.crest_height,
                top_layer_type=20.0,
                previous_top_layer_type=float("nan"),
                top_layer_thickness=float("nan"),
                beta_block_revetment=float("nan"),
                beta_grass_revetment=self._evaluate_grass_revetment_data(
                    evaluation_year
                ),
                reinforce=True,
                tan_alpha=revetment_data_class.slope_parts[-1].end_part,
            )
            _evaluated_measures.append(_extra_measure)

        if transition_level > revetment_data_class.current_transition_level:
            self._correct_revetment_measure_data(_evaluated_measures, transition_level)

        return _evaluated_measures

    def _calculate_revetment_measure_cost(
        self, revetment_measure: RevetmentMeasureData, section_length: float, year: int
    ):
        opslagfactor = 2.509
        discontovoet = 1.02

        # Opnemen en afvoeren oude steenbekleding naar verwerker (incl. stort-/recyclingskosten)
        _cost_remove_steen = 5.49

        # Opnemen en afvoeren teerhoudende oude asfaltbekleding (D=15cm) (incl. stort-/recyclingskosten)
        _cost_remove_asfalt = 13.52

        # Leveren en aanbrengen (verwerken) betonzuilen, incl. doek, vijlaag en inwassen
        D = np.array([0.3, 0.35, 0.4, 0.45, 0.5])
        cost = np.array([72.52, 82.70, 92.56, 102.06, 111.56])
        f = interp1d(D, cost, fill_value=("extrapolate"))
        cost_new_steen = f(revetment_measure.top_layer_thickness)

        _slope_part_difference = (
            revetment_measure.end_part - revetment_measure.begin_part
        )
        x = _slope_part_difference / revetment_measure.tan_alpha

        if x < 0.0 or revetment_measure.end_part < revetment_measure.begin_part:
            raise ValueError("Calculation of design area not possible!")

        # calculate area of new design
        z = np.sqrt(x**2 + _slope_part_difference**2)
        area = z * section_length

        if isinstance(revetment_measure, StoneSlopePart):  # cost of new steen
            cost_vlak = _cost_remove_steen + cost_new_steen
        elif revetment_measure.top_layer_type == 2026.0:
            # cost of new steen, when previous was gras
            cost_vlak = cost_new_steen
        elif GrassSlopePart.is_grass_part(revetment_measure.top_layer_type):
            # cost of removing old revetment when new revetment is gras
            if revetment_measure.previous_top_layer_type == 5.0:
                cost_vlak = _cost_remove_asfalt
            elif revetment_measure.previous_top_layer_type == 20.0:
                cost_vlak = 0.0
            else:
                cost_vlak = _cost_remove_steen
        else:
            cost_vlak = 0.0

        return area * cost_vlak * opslagfactor / discontovoet ** (year - 2025)

    def _calculate_year_costs(
        self,
        revetment_measures: list[RevetmentMeasureData],
        section_length: float,
        year: int,
    ) -> list[float]:
        _costs = []
        for _measure in revetment_measures:
            _costs.append(
                self._calculate_revetment_measure_cost(_measure, section_length, year)
            )
        return _costs

    def evaluate_measure(
        self,
        dike_section: DikeSection,
        traject_info: DikeTrajectInfo,
        preserve_slope: bool,
    ):
        # We are missing the following properties:
        # Then we evaluate
        self.measures = {}

        # Set reliability and cost
        _reliability = self._get_configured_section_reliability(
            dike_section, traject_info
        )
        _reliability.calculate_section_reliability()
        self.measures["Reliability"] = _reliability
        self.measures["Revetment"] = "yes"

    def _evaluate_grass_revetment_data(self, evaluation_year: int) -> float:
        return RevetmentCalculator(self.revetment)._evaluate_grass(evaluation_year)

    def _evaluate_stone_revetment_data(
        self, slope_part: SlopePartProtocol, evaluation_year: int
    ) -> float:
        return RevetmentCalculator(self.revetment)._evaluate_block(
            slope_part, evaluation_year
        )
