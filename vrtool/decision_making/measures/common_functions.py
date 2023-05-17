import copy
import logging
from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely.geometry import Polygon

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


def implement_berm_widening(
    berm_input,
    measure_input,
    measure_parameters,
    mechanism,
    computation_type,
    path_intermediate_stix: Path,
    SFincrease=0.2,
    depth_screen: Optional[float] = None,
):
    """

    Args:
        berm_input (dict): input dictionary of the mechanism
        measure_input (dict): input dictionary of the measure
        measure_parameters (dict): parameters dictionary of the measure
        mechanism (str): name of the mechanism, one of ['Piping', 'Overflow', 'StabilityInner']
        computation_type (str): type of computation for the mechanism
        path_intermediate_stix (Path): path to the intermediate stix files
        SFincrease (float): increase in safety factor
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
                SFincrease,
            )
        )

    # this function implements a berm widening based on the relevant inputs
    if mechanism == "Overflow":
        berm_input["h_crest"] = berm_input["h_crest"] + measure_input["dcrest"]
    elif mechanism == "StabilityInner":
        # Case where the berm widened through DStability and the stability factors will be recalculated
        if computation_type == "DStability":
            _dstability_wrapper = DStabilityWrapper(
                stix_path=Path(berm_input["STIXNAAM"]),
                externals_path=Path(berm_input["DStability_exe_path"]),
            )

            _dstability_berm_widening = BermWideningDStability(
                measure_input=measure_input, dstability_wrapper=_dstability_wrapper
            )

            if measure_input["StabilityScreen"] == "Yes":
                _inner_toe = measure_input["geometry"].loc["BIT"]
                _dstability_wrapper.add_stability_screen(
                    bottom_screen=_inner_toe.z - depth_screen, location=_inner_toe.x
                )

            #  Update the name of the stix file in the mechanism input dictionary, this is the stix that will be used
            # by the calculator later on. In this case, we need to force the wrapper to recalculate the DStability
            # model, hence RERUN_STIX set to True.
            berm_input[
                "STIXNAAM"
            ] = _dstability_berm_widening.create_new_dstability_model(
                path_intermediate_stix
            )
            berm_input["RERUN_STIX"] = True

            return berm_input

        # For stability factors
        if "SF_2025" in berm_input:
            # For now, inward and outward are the same!
            if (measure_parameters["Direction"] == "inward") or (
                measure_parameters["Direction"] == "outward"
            ):
                berm_input["SF_2025"] = berm_input["SF_2025"] + (
                    measure_input["dberm"] * berm_input["dSF/dberm"]
                )
                berm_input["SF_2075"] = berm_input["SF_2075"] + (
                    measure_input["dberm"] * berm_input["dSF/dberm"]
                )
            if measure_parameters["StabilityScreen"] == "yes":
                berm_input["SF_2025"] += SFincrease
                berm_input["SF_2075"] += SFincrease
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
        elif "BETA" in berm_input:
            # TODO remove hard-coded parameter. Should be read from input sheet (the 0.13 in the code)
            berm_input["BETA"] = berm_input["BETA"] + (0.13 * measure_input["dberm"])
            if measure_parameters["StabilityScreen"] == "yes":
                berm_input[
                    "BETA"
                ] = calculate_stability_inner_reliability_with_safety_screen(
                    berm_input["BETA"]
                )
        else:
            raise Exception("Unknown input data for stability when widening the berm")

    elif mechanism == "Piping":
        berm_input["Lvoor"] = berm_input["Lvoor"] + measure_input["dberm"]
        # input['Lachter'] = np.max([0., input['Lachter'] - measure_input['dberm']])
        berm_input["Lachter"] = (berm_input["Lachter"] - measure_input["dberm"]).clip(0)
    return berm_input


def calculate_area(geometry):
    polypoints = []
    for label, points in geometry.iterrows():
        polypoints.append((points.x, points.z))
    polygonXZ = Polygon(polypoints)
    areaPol = Polygon(polygonXZ).area
    return areaPol, polygonXZ


def modify_geometry_input(initial: pd.DataFrame, berm_height: float):
    """Checks geometry and corrects if necessary"""
    # TODO move this to the beginning for the input.
    # modify the old structure
    if not "BUK" in initial.index:
        initial = (
            initial.replace(
                {
                    "innertoe": "BIT",
                    "innerberm1": "EBL",
                    "innerberm2": "BBL",
                    "innercrest": "BIK",
                    "outercrest": "BUK",
                    "outertoe": "BUT",
                }
            )
            .reset_index()
            .set_index("type")
        )

    if initial.loc["BUK"].x != 0.0:
        # if BUK is not at x = 0 , modify entire profile
        initial["x"] = np.subtract(initial["x"], initial.loc["BUK"].x)

    if initial.loc["BUK"].x > initial.loc["BIK"].x:
        # BIK must have larger x than BUK, so likely the profile is mirrored, mirror it back:
        initial["x"] = np.multiply(initial["x"], -1.0)
    # if EBL and BBL not there, generate them.
    if not "EBL" in initial.index:
        inner_slope = np.abs(initial.loc["BIT"].z - initial.loc["BIK"].z) / np.abs(
            initial.loc["BIT"].x - initial.loc["BIK"].x
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
    geometry_plot: bool,
    plot_dir: Union[Path, None] = None,
    berm_height: float = 2,
    slope_in: bool = False,
    crest_extra: float = np.nan,
):
    """initial should be a DataFrame with index values BUT, BUK, BIK, BBL, EBL and BIT.
    If this is not the case and it is input of the old type, first it is transformed to obey that.
    crest_extra is an additional argument in case the crest height for overflow is higher than the BUK and BIT.
    In such cases the crest heightening is the given increment + the difference between crest_extra and the BUK/BIT,
    such that after reinforcement the height is crest_extra + increment.
    It has to be ensured that the BUK has x = 0, and that x increases inward"""
    initial = modify_geometry_input(initial, berm_height)

    # Geometry is always from inner to outer toe
    dcrest = geometry_change[0]
    dberm = geometry_change[1]
    if (~np.isnan(crest_extra)) and crest_extra < initial["z"].max():
        # case where cross section for overflow has a lower spot, but majority of section is higher.
        # in that case the crest height is modified to the level of the overflow computation which is a conservative estimate.
        initial.loc["BIK", "z"] = crest_extra
        initial.loc["BUK", "z"] = crest_extra

    # crest heightening
    if dcrest > 0:
        # determine widening at toes.
        slope_out = np.abs(initial.loc["BUK"].x - initial.loc["BUT"].x) / np.abs(
            initial.loc["BUK"].z - initial.loc["BUT"].z
        )
        BUT_dx = slope_out * dcrest

        # TODO discuss with WSRL: if crest is heightened, should slope be determined based on BIK and BIT or BIK and BBL?
        # Now it has been implemented that the slope is based on BIK and BBL
        slope_in = np.abs(initial.loc["BBL"].x - initial.loc["BIK"].x) / np.abs(
            initial.loc["BBL"].z - initial.loc["BIK"].z
        )
        BIT_dx = slope_in * dcrest
    else:
        BUT_dx = 0.0
        BIT_dx = 0.0

    new_geometry = copy.deepcopy(initial)

    # get effects of inward/outward:
    dhouse = 0.0
    if direction == "outward":
        dout = BUT_dx
        din = BIT_dx
        if dberm <= max_berm_out:
            dhouse = max(0, -(dberm + dout - din))
            shift = dberm
        else:
            berm_in = dberm - max_berm_out
            dhouse = max(0, -(-berm_in + dout - din))
            shift = max_berm_out
    else:
        # all changes inward.
        dhouse = max(0, dberm + BUT_dx + BIT_dx)
        shift = 0.0

    # apply dcrest, dberm and shift due to inward/outward:
    max_to_right = BUT_dx + BIT_dx + dberm - shift
    new_geometry.loc["BUT"].x -= shift
    new_geometry.loc["BUK"].x += BUT_dx - shift
    new_geometry.loc["BIK"].x += BUT_dx - shift
    new_geometry.loc["BBL"].x += BUT_dx + BIT_dx - shift
    new_geometry.loc["EBL"].x += max_to_right
    new_geometry.loc["BIT"].x += max_to_right
    new_geometry.loc["BUK"].z += dcrest
    new_geometry.loc["BIK"].z += dcrest

    # add extra points:
    base = copy.deepcopy(initial)
    initial = add_extra_points(initial, base, (shift, max_to_right))
    new_geometry = add_extra_points(new_geometry, base, (shift, max_to_right))

    # calculate the area difference
    area_old, polygon_old = calculate_area(initial)
    area_new, polygon_new = calculate_area(new_geometry)

    if polygon_old.intersects(polygon_new):  # True
        try:
            poly_intsects = polygon_old.intersection(polygon_new)
        except:
            plt.plot(initial.x, initial.z, "ko")
            plt.plot(*polygon_old.exterior.xy, "g")
            plt.plot(*polygon_new.exterior.xy, "r--")
            plt.savefig("testgeom.png")
            plt.close()
        area_intersect = polygon_old.intersection(polygon_new).area  # 1.0
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

        if geometry_plot:
            if not plot_dir.joinpath("Geometry").is_dir():
                # plot_dir.joinpath.mkdir(parents=True, exist_ok=True)
                plot_dir.joinpath("Geometry").mkdir(parents=True, exist_ok=True)
            plt.plot(initial.loc[:, "x"], initial.loc[:, "z"], "k")
            plt.plot(new_geometry.loc[:, "x"], new_geometry.loc[:, "z"], "--r")
            if poly_diff.area > 0:
                if hasattr(poly_diff, "geoms"):
                    for i in range(len(poly_diff.geoms)):
                        x1, y1 = poly_diff.geoms[i].exterior.xy
                        plt.fill(x1, y1, "r--", alpha=0.1)
                else:
                    x1, y1 = poly_diff.exterior.xy
                    plt.fill(x1, y1, "r--", alpha=0.1)
            if poly_diff2.area > 0:
                if hasattr(poly_diff2, "geoms"):
                    for i in range(len(poly_diff2.geoms)):
                        x1, y1 = poly_diff2.geoms[i].exterior.xy
                        plt.fill(x1, y1, "b--", alpha=0.8)
                else:
                    x1, y1 = poly_diff2.exterior.xy
                    plt.fill(x1, y1, "b--", alpha=0.8)  #
            # if hasattr(poly_intsects, 'geoms'):
            #     for i in range(len(poly_intsects.geoms)):
            #         x1, y1 = poly_intsects[i].exterior.xy
            #         plt.fill(x1, y1, 'g--', alpha=.1)
            # else:
            #     x1, y1 = poly_intsects.exterior.xy
            #     plt.fill(x1, y1, 'g--', alpha=.1)
            # plt.show()

            plt.text(
                np.mean(new_geometry.loc[:, "x"]),
                np.max(new_geometry.loc[:, "z"]),
                "Area extra = {:.4} $m^2$\nArea excavated = {:.4} $m^2$".format(
                    str(area_extra), str(area_excavate)
                ),
            )

            plt.savefig(
                plot_dir.joinpath(
                    "Geometry_" + str(dberm) + "_" + str(dcrest) + direction + ".png"
                )
            )
            plt.close()

    return new_geometry, area_extra, area_excavate, dhouse


# Script to determine the costs of a reinforcement:
def determine_costs(
    parameters,
    type: str,
    length: float,
    unit_costs: dict,
    dcrest: float = 0.0,
    dberm_in: float = 0.0,
    housing: bool = False,
    area_extra: bool = False,
    area_excavated: bool = False,
    direction: bool = False,
    section: str = "",
) -> float:
    if (type == "Soil reinforcement") and (direction == "outward") and (dberm_in > 0.0):
        # as we only use unit costs for outward reinforcement, and these are typically lower, the computation might be incorrect (too low).
        logging.warn(
            "Encountered outward reinforcement with inward berm. Cost computation might be inaccurate"
        )
    if type == "Soil reinforcement":
        if direction == "inward":
            total_cost = (
                unit_costs["Inward added volume"] * area_extra * length
                + unit_costs["Inward starting costs"] * length
            )
        elif direction == "outward":
            volume_excavated = area_excavated * length
            volume_extra = area_extra * length
            reusable_volume = unit_costs["Outward reuse factor"] * volume_excavated
            # excavate and remove part of existing profile:
            total_cost = unit_costs["Outward removed volume"] * (
                volume_excavated - reusable_volume
            )

            # apply reusable volume
            total_cost += unit_costs["Outward reused volume"] * reusable_volume
            remaining_volume = volume_extra - reusable_volume

            # add additional soil:
            total_cost += unit_costs["Outward added volume"] * remaining_volume

            # compensate:
            total_cost += (
                unit_costs["Outward removed volume"]
                * unit_costs["Outward compensation factor"]
                * volume_extra
            )

        else:
            raise Exception("invalid direction")

        # add costs for housing
        if isinstance(housing, pd.DataFrame) and dberm_in > 0.0:
            if dberm_in > housing.size:
                logging.warn(
                    "Inwards reinforcement distance exceeds data for housing database at section {}".format(
                        section
                    )
                )
                # raise Exception('inwards distance exceeds housing database')
                total_cost += (
                    unit_costs["House removal"]
                    * housing.loc[housing.size]["cumulative"]
                )
            else:
                total_cost += (
                    unit_costs["House removal"]
                    * housing.loc[float(dberm_in)]["cumulative"]
                )

        # add costs for stability screen
        if parameters["StabilityScreen"] == "yes":
            total_cost += unit_costs["Sheetpile"] * parameters["Depth"] * length

        if dcrest > 0.0:
            total_cost += unit_costs["Road renewal"] * length

        # x = map(int, self.parameters['house_removal'].split(';'))
    elif type == "Vertical Geotextile":
        total_cost = unit_costs["Vertical Geotextile"] * length
    elif type == "Diaphragm Wall":
        total_cost = unit_costs["Diaphragm wall"] * length
    elif type == "Stability Screen":
        total_cost = unit_costs["Sheetpile"] * parameters["Depth"] * length
    else:
        logging.info("Unknown type")
    return total_cost


# Script to determine the required crest height for a certain year
def probabilistic_design(
    design_variable: str,
    strength_input,
    p_t: float,
    t_0: int,
    horizon: int = 50,
    load_change: float = 0,
    mechanism: str = "Overflow",
    type: str = "SAFE",
) -> float:
    if mechanism == "Overflow":
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
            raise Exception("Unknown calculation type for {}".format(mechanism))
