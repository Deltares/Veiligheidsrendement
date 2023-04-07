import math
from typing import List, Union, Tuple

import numpy as np
from shapely import Polygon, LineString, MultiPolygon, Point, unary_union

def get_modified_meas_geom(soil_measure, straight_line: bool) -> List[Tuple[float, float]]:
    """
    Adapt and modify the geometry of the soil measure and return the surface line as a list of points.

    :param soil_measure: soil measure
    :param straight_line: if True, the geometry is a straight line from the BIT in the hinterland
                          if False, the geometry of the measure is extended
    :return: list of points
    """

    df = soil_measure["Geometry"]
    # EXT and BIT_0 are virtual points and must be removed
    df = df[df.index != "EXT"]
    df = df[df.index != "BIT_0"]

    # Continue the geometry in the hinterland as a straight line
    if straight_line:
        df.loc["BIT_1"] = [100, df.loc["BIT", "z"]]  # this is horizontal line to hinterland2
        return [(x, z) for x, z in zip(df["x"].values, df["z"].values)]

    else:
        ##extend geometry LEFT SIDE
        start = (df.loc["BUK", "x"], df.loc["BUK", "z"])
        end = (df.loc["BUT", "x"], df.loc["BUT", "z"])
        x3, y3 = find_extended_end(start, end, 'left', 10)
        df.loc["left"] = [x3, y3]

        ##extend geometry RIGHT SIDE
        start = (df.loc["EBL", "x"], df.loc["EBL", "z"])
        end = (df.loc["BIT", "x"], df.loc["BIT", "z"])
        x4, y4 = find_extended_end(start, end, 'right', 10)
        df.loc["right"] = [x4, y4]

        # sort df by ascending x:
        df = df.sort_values(by="x")
        return [(x, z) for x, z in zip(df["x"].values, df["z"].values)]


def find_extended_end(start_point: Tuple[float, float], end_point: Tuple[float, float], side: str,
                      desired_length: float) -> Tuple[float, float]:
    """
    Extend a line with a given length in a given direction.

    :param start_point: start point of the line
    :param end_point: end point of the line
    :param side: if the line is extended on the right (hinterland) or left (foreland) of the embankment
    :param desired_length: length of the extension

    :return: new end point of the extended line
    """
    sign = 1 if side == "right" else -1

    m = (end_point[1] - start_point[1]) / (end_point[0] - start_point[0])

    # Calculate the new endpoint of the extended line
    new_x = end_point[0] + sign * (desired_length / math.sqrt(1 + m ** 2))
    new_y = end_point[1] + m * (new_x - end_point[0])
    return new_x, new_y

def crop_polygons_below_surface_line(polygons: List[Polygon], top_surface: np.array) -> List[Polygon]:
    """
    Crop all the polygons below the new line of the top surface from the measure.

    :param polygon_list: list of polygons from the current situation (imported from stix file)
    :param top_surface: list of points of the new top surface
    :return: list of polygons cropped below the new top surface
    """
    new_line = LineString([Point(i) for i in top_surface])
    polygon_above_surface = Polygon([pt for pt in new_line.coords] + [(100, 30), (-100, 30)])
    new_polygon_collection = []
    for polygon in polygons:
        if not polygon.intersects(new_line):
            new_polygon_collection.append(polygon)
        else:

            polygon = polygon.difference(polygon_above_surface)
            new_polygon_collection.append(polygon)

    return new_polygon_collection


def find_polygons_to_fill_to_measure(polygons: List[Polygon], top_surface: np.array) -> List[Polygon]:
    """
    Find all the polygons that fill the gap between the original collection of polygons from a stix file and the new
    line of the top surface from the measure.
    All portions of the original polygons lying above the new top surface are removed.

    :param polygon_list: list of polygons from the current situation (imported from a .stix file)
    :param top_surface: The (new) top surface line of the measure for which the polygons should be cropped to
    :return: The updated polygon list
    """

    # create two polygons that lie above and below the surface line to + infinity (=30) and - infinity (=-50)
    new_line = LineString([Point(i) for i in top_surface])
    polygon_above_surface = Polygon([pt for pt in new_line.coords] + [(100, 30), (-100, 30)])
    polygon_below_surface = Polygon([pt for pt in new_line.coords] + [(100, -50), (-100, -50)])

    # merge all input polygons together
    merged_polygon = unary_union(polygons)

    # Do the difference between the merged polygon with the polygon lying above the surface line.
    # This returns an intermediate cropped polygon of the merged polygon that is below the surface line.
    cropped_polygon_below_surface = merged_polygon.difference(polygon_above_surface)

    # Now do the difference between the merged polygon with the polygon lying below the surface line.
    #  This returns one or several polygons.
    diff_poly_and_surface = polygon_below_surface.difference(merged_polygon)

    # Get the polygons that will fill the gap between the merged polygon and the surface line.
    filling_polygons = get_filling_polygons(diff_poly_and_surface, top_surface, cropped_polygon_below_surface)

    return filling_polygons


def get_filling_polygons(diff_poly_and_surface: Union[MultiPolygon, Polygon],
                         top_surface: np.array,
                         cropped_polygon_below_surface: Polygon) -> List[Polygon]:
    """
    Get the polygons that need to be filled to the top surface

    :param diff_poly_and_surface: The difference between the merged polygon and the polygon lying below the surface line
    :param top_surface: The geometry of measure's surface line
    :param cropped_polygon_below_surface: The cropped polygon that lies below the top surface line
    """
    keep_poly = []

    # Find the intersection between the bounding box and the polygon
    bbox = get_bounding_box(top_surface, cropped_polygon_below_surface)
    intersection = diff_poly_and_surface.intersection(bbox)

    if isinstance(intersection, Polygon):  # if the intersection is a polygon, add it to the list directly
        keep_poly.append(intersection)
    else:
        for intersected_polygon in intersection.geoms:
            if isinstance(intersected_polygon, Polygon):  # eliminate weird linestrings
                keep_poly.append(intersected_polygon)
    return keep_poly


def get_bounding_box(top_surface: np.array, cropped_polygon_below_surface: Polygon) -> Polygon:
    """
    Get the bounding box of the polygon that is below the top surface line

    :param top_surface: The geometry of measure's surface line
    :param cropped_polygon_below_surface: The cropped polygon that lies below the top surface line
    :return:
    """
    min_x, min_y, max_x, _ = cropped_polygon_below_surface.bounds
    max_y = max([p[1] for p in top_surface])  # the max of the bounding box should be the top of the measure geometry
    bounding_box = Polygon([(min_x, min_y), (min_x, max_y), (max_x, max_y), (max_x, min_y)])

    return bounding_box
