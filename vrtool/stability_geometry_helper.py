from typing import List, Union

import numpy as np
from shapely import Polygon, LineString, MultiPolygon, Point, unary_union


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
