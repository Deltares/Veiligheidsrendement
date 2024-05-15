import copy
import logging
import math
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from shapely.geometry import Polygon

from vrtool.common.enums.measure_type_enum import MeasureTypeEnum
from vrtool.common.enums.mechanism_enum import MechanismEnum
from vrtool.common.measure_unit_costs import MeasureUnitCosts
from vrtool.decision_making.measures.berm_widening_dstability import (
    BermWideningDStability,
)
from vrtool.failure_mechanisms.overflow.overflow_functions import (
    calculate_overflow_hydra_ring_design,
    calculate_overflow_simple_design,
)
from vrtool.failure_mechanisms.stability_inner.dstability_wrapper import (
    DStabilityWrapper,
)
from vrtool.failure_mechanisms.stability_inner.stability_inner_functions import (
    calculate_reliability,
    calculate_safety_factor,
)
from vrtool.flood_defence_system.dike_section import DikeSection


def implement_berm_widening(
    berm_input,
    measure_input,
    measure_parameters,
    mechanism: MechanismEnum,
    computation_type,
    is_first_year_with_widening: bool,
    path_intermediate_stix: Path,
    depth_screen: Optional[float] = None,
):
    """

    Args:
        berm_input (dict): input dictionary of the mechanism
        measure_input (dict): input dictionary of the measure
        measure_parameters (dict): parameters dictionary of the measure
        mechanism (MechanismEnum): mechanism, one of [MechanismEnum.PIPING, MechanismEnum.OVERFLOW, MechanismEnum.STABILITY_INNER]
        computation_type (str): type of computation for the mechanism
        is_first_year_with_widening (bool): flag for triggering rerunning stix
        path_intermediate_stix (Path): path to the intermediate stix files
        depth_screen (float): depth of the stability screen

    Returns:
        dict: input dictionary of the mechanism with the berm widened

    """

    def calculate_stability_inner_reliability_with_safety_screen(
        reliability: np.ndarray,
    ):
        # convert to SF and back:
        return calculate_reliability(
            np.add(
                calculate_safety_factor(reliability),
                _safety_factor_increase,
            )
        )

    _safety_factor_increase = get_safety_factor_increase(measure_input["l_stab_screen"])

    # this function implements a berm widening based on the relevant inputs
    if mechanism == MechanismEnum.OVERFLOW:
        berm_input["h_crest"] = berm_input["h_crest"] + measure_input["dcrest"]
    elif mechanism == MechanismEnum.STABILITY_INNER:
        # Case where the berm widened through DStability and the stability factors will be recalculated
        if computation_type.lower() == "dstability":
            _dstability_wrapper = DStabilityWrapper(
                stix_path=Path(berm_input["STIXNAAM"]),
                externals_path=Path(berm_input["DStability_exe_path"]),
            )

            _dstability_berm_widening = BermWideningDStability(
                measure_input=measure_input, dstability_wrapper=_dstability_wrapper
            )
            if measure_input["StabilityScreen"].lower().strip() == "yes":
                _inner_toe = measure_input["geometry"].loc["BIT"]
                _dstability_wrapper.add_stability_screen(
                    bottom_screen=_inner_toe.z - depth_screen, location=_inner_toe.x
                )

            #  Update the name of the stix file in the mechanism input dictionary, this is the stix that will be used
            # by the calculator later on. In this case, we need to force the wrapper to recalculate the DStability
            # model, hence RERUN_STIX set to True, but only for the investment year.
            berm_input[
                "STIXNAAM"
            ] = _dstability_berm_widening.create_new_dstability_model(
                path_intermediate_stix
            )
            if is_first_year_with_widening:
                berm_input["RERUN_STIX"] = True

            return berm_input

        # For stability factors
        if "sf_2025" in berm_input:
            # For now, inward and outward are the same!
            if (measure_parameters["Direction"] == "inward") or (
                measure_parameters["Direction"] == "outward"
            ):
                berm_input["sf_2025"] = berm_input["sf_2025"] + (
                    measure_input["dberm"] * berm_input["dSF/dberm"]
                )
                berm_input["sf_2075"] = berm_input["sf_2075"] + (
                    measure_input["dberm"] * berm_input["dSF/dberm"]
                )
            if measure_parameters["StabilityScreen"] == "yes":
                berm_input["sf_2025"] += _safety_factor_increase
                berm_input["sf_2075"] += _safety_factor_increase
        # For betas as input
        elif "beta_2025" in berm_input:
            berm_input["beta_2025"] = berm_input["beta_2025"] + (
                measure_input["dberm"] * berm_input["dbeta/dberm"]
            )
            berm_input["beta_2075"] = berm_input["beta_2075"] + (
                measure_input["dberm"] * berm_input["dbeta/dberm"]
            )
            if measure_parameters["StabilityScreen"] == "yes":
                berm_input[
                    "beta_2025"
                ] = calculate_stability_inner_reliability_with_safety_screen(
                    berm_input["beta_2025"]
                )
                berm_input[
                    "beta_2075"
                ] = calculate_stability_inner_reliability_with_safety_screen(
                    berm_input["beta_2075"]
                )
        elif "beta" in berm_input:
            # TODO remove hard-coded parameter. Should be read from input sheet (the 0.13 in the code)
            berm_input["beta"] = berm_input["beta"] + (0.13 * measure_input["dberm"])
            if measure_parameters["StabilityScreen"] == "yes":
                berm_input[
                    "beta"
                ] = calculate_stability_inner_reliability_with_safety_screen(
                    berm_input["beta"]
                )
        else:
            raise NotImplementedError(
                "Unknown input data for stability when widening the berm, available: {}".format(
                    ",".join(berm_input.keys())
                )
            )

    elif mechanism == MechanismEnum.PIPING:
        berm_input["l_voor"] = berm_input["l_voor"] + measure_input["dberm"]
        # input['Lachter'] = np.max([0., input['Lachter'] - measure_input['dberm']])
        berm_input["l_achter"] = (berm_input["l_achter"] - measure_input["dberm"]).clip(
            0
        )
        if measure_parameters["StabilityScreen"] == "yes":
            berm_input["sf_factor"] = sf_factor_piping(measure_input["l_stab_screen"])
    return berm_input

def get_safety_factor_increase(l_stab_screen: float) -> float:
    """
    get the safety factor for stability that now depends on the length of the stability screen

    Args:
        l_stab_screen (float): length of the screen (without cover layer thickness)

    Returns:
        float: safe factor increase; 0.2 for 3m and 0.4 for 6m
    """
    _default_safety_factor = 0.2
    _small_stab_screen = 3.0
    if math.isnan(l_stab_screen):
        return _default_safety_factor
    return _default_safety_factor * l_stab_screen / _small_stab_screen

def sf_factor_piping(length: float) -> float:
    """
    get the safe reduction factor for the probability of piping

    Args:
        length (float): length of the screen (without cover layer thickness)

    Returns:
        float: the safe reduction factor: 100 for 3m; 1000 for 6m
    """
    _small_stab_screen = 3.0
    return 10 ** (1.0 + length / _small_stab_screen)

def calculate_area(geometry):
    polypoints = []
    for _, points in geometry.iterrows():
        polypoints.append((points.x, points.z))
    polygonXZ = Polygon(polypoints)
    areaPol = Polygon(polygonXZ).area
    return areaPol, polygonXZ


def modify_geometry_input(initial: pd.DataFrame, berm_height: float) -> pd.DataFrame:
    """Checks geometry and corrects if necessary"""

    if initial.loc["BUK"].x != 0.0:
        # if BUK is not at x = 0 , modify entire profile:
        initial["x"] = np.subtract(initial["x"], initial.loc["BUK"].x)

    if initial.loc["BUK"].x > initial.loc["BIK"].x:
        # BIK must have larger x than BUK, so likely the profile is mirrored, mirror it back:
        initial["x"] = np.multiply(initial["x"], -1.0)

    if not "EBL" in initial.index:
        # if EBL and BBL are not there, generate them:
        inner_slope = np.abs(initial.loc["BIT"].z - initial.loc["BIK"].z) / np.abs(
            initial.loc["BIT"].x - initial.loc["BIK"].x
        )
        berm_height = min(
            berm_height, initial.loc["BIK"].z - initial.loc["BIT"].z - 0.01
        )
        initial.loc["EBL", "x"] = initial.loc["BIT"].x - (berm_height / inner_slope)
        initial.loc["BBL", "x"] = initial.loc["BIT"].x - (berm_height / inner_slope)
        initial.loc["BBL", "z"] = initial.loc["BIT"].z + berm_height
        initial.loc["EBL", "z"] = initial.loc["BIT"].z + berm_height
        initial = initial.reindex(["BUT", "BUK", "BIK", "BBL", "EBL", "BIT"])

    return initial


def add_extra_points(
    geom: pd.DataFrame, base: pd.DataFrame, to_left_right: tuple[float, float]
) -> pd.DataFrame:
    dxL = 1.0 + to_left_right[0]
    dxR = 1.0 + to_left_right[1]
    dz = 1.0

    ltp = [base.loc["BUT"].x - dxL, base.loc["BUT"].z]
    lowestz = min(base.loc["BUT"].z, base.loc["BIT"].z) - dz
    lbp = [base.loc["BUT"].x - dxL, lowestz]
    rtp = [base.loc["BIT"].x + dxR, base.loc["BIT"].z]
    rbp = [base.loc["BIT"].x + dxR, lowestz]

    geom.loc["LBT"] = pd.Series(lbp, index=["x", "z"])
    geom.loc["LTP"] = pd.Series(ltp, index=["x", "z"])
    geom.loc["RTP"] = pd.Series(rtp, index=["x", "z"])
    geom.loc["RBP"] = pd.Series(rbp, index=["x", "z"])

    geom = geom.reindex(
        ["LBT", "LTP", "BUT", "BUK", "BIK", "BBL", "EBL", "BIT", "RTP", "RBP"]
    )
    return geom


# This script determines the new geometry for a soil reinforcement based on a 4 or 6 point profile
def determine_new_geometry(
    geometry_change: tuple[float, float],
    direction: float,
    max_berm_out: float,
    initial: pd.DataFrame,
    berm_height: float = 2,
    crest_extra: float = np.nan,
) -> list:
    """initial should be a DataFrame with index values BUT, BUK, BIK, BBL, EBL and BIT.
    If this is not the case, first it is transformed to obey that.
    crest_extra is an additional argument in case the crest height for overflow is higher than the BUK and BIT.
    In such cases the crest heightening is the given increment + the difference between crest_extra and the BUK/BIT,
    such that after reinforcement the height is crest_extra + increment.
    It has to be ensured that the BUK has x = 0, and that x increases inward

    Returns:
        four values: new_geometry, area_extra, area_excavate, d_house
    Raises:
        ValueError if intersection of geometries fails
    """

    initial = modify_geometry_input(initial, berm_height)

    # Geometry is always from inner to outer toe
    _d_crest = geometry_change[0]
    _d_berm = geometry_change[1]
    if (~np.isnan(crest_extra)) and crest_extra < initial["z"].max():
        # case where cross section for overflow has a lower spot, but majority of section is higher.
        # in that case the crest height is modified to the level of the overflow computation which is a conservative estimate.
        initial.loc["BIK", "z"] = crest_extra
        initial.loc["BUK", "z"] = crest_extra

    # crest heightening
    if _d_crest > 0:
        # determine widening at toes.
        slope_out = np.abs(initial.loc["BUK"].x - initial.loc["BUT"].x) / np.abs(
            initial.loc["BUK"].z - initial.loc["BUT"].z
        )
        _but_dx = slope_out * _d_crest

        # TODO discuss with WSRL: if crest is heightened, should slope be determined based on BIK and BIT or BIK and BBL?
        # Now it has been implemented that the slope is based on BIK and BBL
        slope_in = np.abs(initial.loc["BBL"].x - initial.loc["BIK"].x) / np.abs(
            initial.loc["BBL"].z - initial.loc["BIK"].z
        )
        _bit_dx = slope_in * _d_crest
    else:
        _but_dx = 0.0
        _bit_dx = 0.0

    _new_geometry = copy.deepcopy(initial)

    # get effects of inward/outward:
    _d_house = 0.0
    if direction == "outward":
        _d_out = _but_dx
        _d_in = _bit_dx
        if _d_berm <= max_berm_out:
            _d_house = max(0, -(_d_berm + _d_out - _d_in))
            shift = _d_berm
        else:
            berm_in = _d_berm - max_berm_out
            _d_house = max(0, -(-berm_in + _d_out - _d_in))
            shift = max_berm_out
    else:
        # all changes inward.
        _d_house = max(0, _d_berm + _but_dx + _bit_dx)
        shift = 0.0

    # apply dcrest, dberm and shift due to inward/outward:
    max_to_right = _but_dx + _bit_dx + _d_berm - shift
    _new_geometry.loc["BUT", "x"] -= shift
    _new_geometry.loc["BUK", "x"] += _but_dx - shift
    _new_geometry.loc["BIK", "x"] += _but_dx - shift
    _new_geometry.loc["BBL", "x"] += _but_dx + _bit_dx - shift
    _new_geometry.loc["EBL", "x"] += max_to_right
    _new_geometry.loc["BIT", "x"] += max_to_right
    _new_geometry.loc["BUK", "z"] += _d_crest
    _new_geometry.loc["BIK", "z"] += _d_crest

    # add extra points:
    base = copy.deepcopy(initial)
    initial = add_extra_points(initial, base, (shift, max_to_right))
    _new_geometry = add_extra_points(_new_geometry, base, (shift, max_to_right))

    # calculate the area difference
    area_old, polygon_old = calculate_area(initial)
    area_new, polygon_new = calculate_area(_new_geometry)

    if polygon_old.intersects(polygon_new):  # True
        try:
            poly_intsects = polygon_old.intersection(polygon_new)
        except:
            raise ValueError(
                "invalid geometry; intersection between original and modified geometry can not be evaluated."
            )
        area_intersect = poly_intsects.area
        area_excavate = area_old - area_intersect
        area_extra = area_new - area_intersect

        # difference new-old = extra
        poly_diff = polygon_new.difference(polygon_old)
        area_diff = poly_diff.area  # zou zelfde moeten zijn als area_extra
        # difference new-old = excavate
        poly_diff2 = polygon_old.difference(polygon_new)
        area_diff2 = poly_diff2.area  # zou zelfde moeten zijn als area_excavate

        # controle
        test1 = area_diff - area_extra
        test2 = area_diff2 - area_excavate
        if test1 > 1 or test2 > 1:
            raise Exception("area calculation failed")

    return _new_geometry, area_extra, area_excavate, _d_house


# Script to determine the costs of a reinforcement:
def determine_costs(
    measure_type: str,
    length: float,
    unit_costs: MeasureUnitCosts,
    dcrest: float = 0.0,
    dberm_in: float = 0.0,
    housing: bool = False,
    area_extra: bool = False,
    area_excavated: bool = False,
    direction: bool = False,
    section: str = "",
) -> float:
    """
    Determine costs, mainly for soil reinforcement
    """
    _measure_type_name = measure_type.lower().strip()
    if (
        (measure_type == MeasureTypeEnum.SOIL_REINFORCEMENT.legacy_name)
        and (direction == "outward")
        and (dberm_in > 0.0)
    ):
        # as we only use unit costs for outward reinforcement, and these are typically lower, the computation might be incorrect (too low).
        logging.warning(
            "Buitenwaartse versterking met binnenwaartse berm (dijkvak {}) kan leiden tot onnauwkeurige kostenberekeningen".format(
                section
            )
        )

    if measure_type in (
        MeasureTypeEnum.SOIL_REINFORCEMENT.legacy_name,
        MeasureTypeEnum.SOIL_REINFORCEMENT_WITH_STABILITY_SCREEN.legacy_name,
    ):
        if direction == "inward":
            total_cost = (
                unit_costs.inward_added_volume * area_extra * length
                + unit_costs.inward_starting_costs * length
            )
        elif direction == "outward":
            volume_excavated = area_excavated * length
            volume_extra = area_extra * length
            reusable_volume = unit_costs.outward_reuse_factor * volume_excavated
            # excavate and remove part of existing profile:
            total_cost = unit_costs.outward_removed_volume * (
                volume_excavated - reusable_volume
            )

            # apply reusable volume
            total_cost += unit_costs.outward_reused_volume * reusable_volume
            remaining_volume = volume_extra - reusable_volume

            # add additional soil:
            total_cost += unit_costs.outward_added_volume * remaining_volume

            # compensate:
            total_cost += (
                unit_costs.outward_removed_volume
                * unit_costs.outward_compensation_factor
                * volume_extra
            )

        else:
            raise ValueError("invalid direction")

        # add costs for housing
        if isinstance(housing, pd.DataFrame) and dberm_in > 0.0:
            if dberm_in > housing.size:
                logging.warning(
                    "Binnenwaartse teenverschuiving is groter dan gegevens voor bebouwing op dijkvak {}".format(
                        section
                    )
                )
                total_cost += (
                    unit_costs.house_removal * housing.loc[housing.size]["cumulative"]
                )
            else:
                total_cost += (
                    unit_costs.house_removal
                    * housing.loc[float(dberm_in)]["cumulative"]
                )

        if dcrest > 0.0:
            total_cost += unit_costs.road_renewal * length

        # x = map(int, self.parameters['house_removal'].split(';'))
    elif measure_type == MeasureTypeEnum.VERTICAL_PIPING_SOLUTION.legacy_name:
        total_cost = unit_costs.vertical_geotextile * length
    elif measure_type == MeasureTypeEnum.DIAPHRAGM_WALL.legacy_name:
        total_cost = unit_costs.diaphragm_wall * length
    else:
        logging.error("Onbekend maatregeltype: {}".format(measure_type))
        total_cost = float("nan")
    return total_cost


# Script to determine the required crest height for a certain year
def probabilistic_design(
    design_variable: str,
    strength_input,
    p_t: float,
    t_0: int,
    horizon: int = 50,
    load_change: float = 0,
    mechanism: MechanismEnum = MechanismEnum.OVERFLOW,
    type: str = "SAFE",
) -> float:
    if mechanism == MechanismEnum.OVERFLOW:
        if type == "SAFE":
            # determine the crest required for the target
            h_crest, beta = calculate_overflow_simple_design(
                strength_input["q_crest"],
                strength_input["h_c"],
                strength_input["q_c"],
                strength_input["beta"],
                failure_probability=p_t,
                design_variable=design_variable,
            )
            # add temporal changes due to settlement and climate change
            h_crest = h_crest + horizon * (strength_input["dhc(t)"] + load_change)
            return h_crest
        elif type == "HRING":
            h_crest, beta = calculate_overflow_hydra_ring_design(
                strength_input, horizon, t_0, p_t
            )
            return h_crest
        else:
            raise ValueError("Unknown calculation type for {}".format(mechanism))
