import copy
import warnings
from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely.geometry import Polygon

from vrtool.failure_mechanisms.overflow.overflow_functions import (
    calculate_overflow_hydra_ring_design,
    calculate_overflow_simple_design,
)
from vrtool.failure_mechanisms.stability_inner.stability_inner_functions import (
    calculate_reliability,
    calculate_safety_factor,
)
import logging


def implement_berm_widening(
    input,
    measure_input,
    measure_parameters,
    mechanism,
    computation_type,
    SFincrease=0.2,
):
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
        input["h_crest"] = input["h_crest"] + measure_input["dcrest"]
    elif mechanism == "StabilityInner":
        # For stability factors
        if "SF_2025" in input:
            # For now, inward and outward are the same!
            if (measure_parameters["Direction"] == "inward") or (
                measure_parameters["Direction"] == "outward"
            ):
                input["SF_2025"] = input["SF_2025"] + (
                    measure_input["dberm"] * input["dSF/dberm"]
                )
                input["SF_2075"] = input["SF_2075"] + (
                    measure_input["dberm"] * input["dSF/dberm"]
                )
            if measure_parameters["StabilityScreen"] == "yes":
                input["SF_2025"] += SFincrease
                input["SF_2075"] += SFincrease
        # For betas as input
        elif "beta_2025" in input:
            input["beta_2025"] = input["beta_2025"] + (
                measure_input["dberm"] * input["dbeta/dberm"]
            )
            input["beta_2075"] = input["beta_2075"] + (
                measure_input["dberm"] * input["dbeta/dberm"]
            )
            if measure_parameters["StabilityScreen"] == "yes":
                input[
                    "beta_2025"
                ] = calculate_stability_inner_reliability_with_safety_screen(
                    input["beta_2025"]
                )
                input[
                    "beta_2075"
                ] = calculate_stability_inner_reliability_with_safety_screen(
                    input["beta_2075"]
                )
        elif "BETA" in input:
            # TODO remove hard-coded parameter. Should be read from input sheet (the 0.13 in the code)
            input["BETA"] = input["BETA"] + (0.13 * measure_input["dberm"])
            if measure_parameters["StabilityScreen"] == "yes":
                input[
                    "BETA"
                ] = calculate_stability_inner_reliability_with_safety_screen(
                    input["BETA"]
                )
        else:
            raise Exception("Unknown input data for stability when widening the berm")

    elif mechanism == "Piping":
        input["Lvoor"] = input["Lvoor"] + measure_input["dberm"]
        # input['Lachter'] = np.max([0., input['Lachter'] - measure_input['dberm']])
        input["Lachter"] = (input["Lachter"] - measure_input["dberm"]).clip(0)
    return input


def calculate_area(geometry):
    polypoints = []
    for label, points in geometry.iterrows():
        polypoints.append((points.x, points.z))
    polygonXZ = Polygon(polypoints)
    areaPol = Polygon(polygonXZ).area
    return areaPol, polygonXZ


def modify_geometry_input(initial, berm_height):
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

    return initial


# This script determines the new geometry for a soil reinforcement based on a 4 or 6 point profile
def determine_new_geometry(
    geometry_change,
    direction,
    max_berm_out,
    initial,
    geometry_plot: bool,
    plot_dir: Union[Path, None] = None,
    berm_height: float = 2,
    slope_in: bool = False,
    crest_extra: bool = False,
):
    """initial should be a DataFrame with index values BUT, BUK, BIK, BBL, EBL and BIT.
    If this is not the case and it is input of the old type, first it is transformed to obey that.
    crest_extra is an additional argument in case the crest height for overflow is higher than the BUK and BIT.
    In such cases the crest heightening is the given increment + the difference between crest_extra and the BUK/BIT,
    such that after reinforcement the height is crest_extra + increment.
    It has to be ensured that the BUK has x = 0, and that x increases inward"""
    initial = modify_geometry_input(initial, berm_height)
    # maxBermOut=20
    # if len(initial) == 6:
    #     noberm = False
    # elif len(initial) == 4:
    #     noberm=True
    # else:
    #     raise Exception ('input length dike is not 4 or 6')

    # if z innertoe != z outertoe add a point to ensure correct shapely operations
    initial.loc["EXT", "x"] = initial.loc["BIK"].x
    initial.loc["EXT", "z"] = np.min(initial.z)

    if initial.loc["BIT"].z > initial.loc["BUT"].z:
        initial.loc["BIT_0", "x"] = initial.loc["BIT"].x
        initial.loc["BIT_0", "z"] = initial.loc["BIT"].z
        initial = initial.reindex(
            ["BUT", "BUK", "BIK", "BBL", "EBL", "BIT", "BIT_0", "EXT"]
        )
    elif initial.loc["BIT"].z < initial.loc["BUT"].z:
        initial.loc["BUT_0", "x"] = initial.loc["BUT"].x
        initial.loc["BUT_0", "z"] = initial.loc["BUT"].z
        initial = initial.reindex(
            ["BUT", "BUT_0", "BUK", "BIK", "BBL", "EBL", "BIT", "EXT"]
        )

    # Geometry is always from inner to outer toe
    dcrest = geometry_change[0]
    dberm = geometry_change[1]
    if crest_extra:
        if (crest_extra > initial["z"].max()) & (dcrest > 0.0):
            # if overflow crest is higher than profile, in case of reinforcement ensure that everything is heightened to that level + increment:
            pass
        elif crest_extra < initial["z"].max():
            # case where cross section for overflow has a lower spot, but majority of section is higher.
            # in that case the crest height is modified to the level of the overflow computation which is a conservative estimate.
            initial.loc["BIK", "z"] = crest_extra
            initial.loc["BUK", "z"] = crest_extra
        cur_crest = crest_extra

    else:
        cur_crest = initial["z"].max()
    new_crest = cur_crest + dcrest

    # crest heightening
    if dcrest > 0:
        # determine widening at toes.
        slope_out = np.abs(initial.loc["BUK"].x - initial.loc["BUT"].x) / np.abs(
            initial.loc["BUK"].z - initial.loc["BUT"].z
        )
        BUT_dx = out = slope_out * dcrest

        # TODO discuss with WSRL: if crest is heightened, should slope be determined based on BIK and BIT or BIK and BBL?
        # Now it has been implemented that the slope is based on BIK and BBL
        slope_in = np.abs(initial.loc["BBL"].x - initial.loc["BIK"].x) / np.abs(
            initial.loc["BBL"].z - initial.loc["BIK"].z
        )
        BIT_dx = slope_in * dcrest
    else:
        BUT_dx = 0.0
        BIT_dx = 0.0
    # z_innertoe = (initial.z[int(initial[initial.type == 'innertoe'].index.values)])

    dhouse = 0.0
    if direction == "outward":
        # nieuwe opzet:
        # if outward:
        #    verplaats buitenkruin en buitenteen
        #   ->tussen geometrie 1
        #  afgraven
        # ->tussen geometrie 2

        # berm er aan plakken. Ook bij alleen binnenwaarts

        # volumes berekenen (totaal extra, en totaal "verplaatst in profiel")

        # optional extension: optimize amount of outward/inward reinforcement
        new_geometry = copy.deepcopy(initial)

        dout = BUT_dx
        din = BIT_dx
        if dberm <= max_berm_out:

            for ind, data in new_geometry.iterrows():
                # Run over points
                if ind in ["EXT", "BUT_0", "BIT_0"]:
                    xz = data.values
                elif ind == "BIT":
                    xz = [data.x + dberm + dout - din, data.z]
                    dhouse = max(0, -(dberm + dout - din))
                elif ind in ["BBL", "EBL"]:
                    xz = [data.x + dberm + dout - din, data.z]
                elif ind in ["BIK", "BUK"]:
                    xz = [data.x + dberm + dout, data.z + dcrest]
                elif ind == "BUT":
                    xz = [data.x + dberm, data.z]
                new_geometry.loc[ind] = pd.Series(xz, index=["x", "z"])

        else:
            berm_in = dberm - max_berm_out
            for ind, data in new_geometry.iterrows():
                # Run over points
                if ind in ["EXT", "BUT_0", "BIT_0"]:
                    xz = data.values
                elif ind == "BIT":
                    xz = [data.x - berm_in + dout - din, data.z]
                    dhouse = max(0, -(-berm_in + dout - din))
                elif ind == "BBL":
                    xz = [data.x - berm_in + dout - din, data.z]
                elif ind == "EBL":
                    xz = [data.x + max_berm_out + dout - din, data.z]
                elif ind in ["BIK", "BUK"]:
                    xz = [data.x + max_berm_out + dout, data.z + dcrest]
                elif ind == "BUT":
                    xz = [data.x + max_berm_out, data.z]
                new_geometry.loc[ind] = pd.Series(xz, index=["x", "z"])

            # if noberm:  # len(initial) == 4:
            #     if dberm > 0:
            #         new_geometry = addBerm(initial, geometry, new_geometry, bermheight, dberm)
            # if (initial.type == 'extra').any():
            #     if dberm > 0 or dcrest > 0:
            #         new_geometry = addExtra(initial, new_geometry)

    if direction == "inward":
        # all changes inward.
        new_geometry = copy.deepcopy(initial)
        for ind, data in new_geometry.iterrows():
            # Run over points .
            if ind in ["EXT", "BUT", "BUT_0", "BIT_0"]:  # Points that are not modified
                xz = data.values
            if ind == "BIT":
                xz = [data.x + dberm + BUT_dx + BIT_dx, data.z]
                dhouse = max(0, dberm + BUT_dx + BIT_dx)
            elif ind == "EBL":
                xz = [data.x + dberm + BUT_dx + BIT_dx, data.z]
            elif ind == "BBL":
                xz = [data.x + BUT_dx + BIT_dx, data.z]
            elif ind == "BIK":
                xz = [data.x + BUT_dx, data.z + dcrest]
            elif ind == "BUK":
                xz = [data.x + BUT_dx, data.z + dcrest]
            new_geometry.loc[ind] = pd.Series(xz, index=["x", "z"])

    # calculate the area difference
    area_old, polygon_old = calculate_area(initial)
    area_new, polygon_new = calculate_area(new_geometry)

    #
    # plt.plot(initial.x,initial.z, 'ko')
    # plt.plot(*polygon_old.exterior.xy, 'g')
    # plt.plot(*polygon_new.exterior.xy, 'r--')
    # plt.savefig('testgeom.png')
    # plt.close()
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
            plt.plot(geometry[:, 0], geometry[:, 1], "k")
            plt.plot(new_geometry[:, 0], new_geometry[:, 1], "--r")
            if poly_diff.area > 0:
                if hasattr(poly_diff, "geoms"):
                    for i in range(len(poly_diff.geoms)):
                        x1, y1 = poly_diff[i].exterior.xy
                        plt.fill(x1, y1, "r--", alpha=0.1)
                else:
                    x1, y1 = poly_diff.exterior.xy
                    plt.fill(x1, y1, "r--", alpha=0.1)
            if poly_diff2.area > 0:
                if hasattr(poly_diff2, "geoms"):
                    for i in range(len(poly_diff2.geoms)):
                        x1, y1 = poly_diff2[i].exterior.xy
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
                np.mean(new_geometry[:, 0]),
                np.max(new_geometry[:, 1]),
                "Area extra = {:.4} $m^2$, area excavated = {:.4} $m^2$".format(
                    str(area_extra), str(area_excavate)
                ),
            )

            plt.savefig(
                plot_dir.joinpath(
                    "Geometry_" + str(dberm) + "_" + str(dcrest) + direction + ".png"
                )
            )
            plt.close()

    area_difference = np.max([0.0, area_extra + 0.5 * area_excavate])
    # old:
    # return new_geometry, area_difference
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
                warnings.warn(
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
