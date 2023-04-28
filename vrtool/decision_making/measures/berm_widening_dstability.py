import math
from pathlib import Path
from typing import Tuple, Union, Optional

import numpy as np
from geolib.geometry import Point as GeolibPoint
from geolib.models.dstability.internal import PersistablePoint, PersistableLayer
from shapely import LineString, Polygon, unary_union, MultiPolygon, Point

from vrtool.failure_mechanisms.stability_inner.dstability_wrapper import DStabilityWrapper


class BermWideningDStability:
    def __init__(self, measure_input: dict, dstability_wrapper: DStabilityWrapper):
        self.geometry = measure_input['Geometry']
        self.dberm = measure_input['dberm']
        self.dcrest = measure_input['dcrest']
        self._measure_geometry_points = None
        self._dstability_wrapper = dstability_wrapper

    measure_geometry_points: list


    def create_new_dstability_model(self, path_intermediate_stix: Path) -> Path:
        """
        Find the new geometry of the dstability model to account for the berm widening measure and create a new stix
        stix file accordingly.

        Args:
            path_intermediate_stix: path to the intermediate stix file

        """
        # Get all the polygons from the stix file.
        # It is assumed that all the stages share the same surface line, so we only need to find the filling polygons for
        # the first stage. If False, then this routine must be rerun for every stage.
        _collection_polygon = [Polygon([(p.X, p.Z) for p in layer.Points]) for layer in
                               self._dstability_wrapper._dstability_model.datastructure.geometries[0].Layers]
        self.measure_geometry_points = self.get_modified_meas_geom(straight_line=False)


        # 1. Run first routine to find the polygons to fill.
        _fill_polygons = self.find_polygons_to_fill_to_measure(_collection_polygon)

        # 2. Run second routine to apply the measure to the dstability model.
        _new_stix_name = self.apply_measure_to_dstability(_fill_polygons, path_intermediate_stix)

        return path_intermediate_stix / _new_stix_name

    def get_modified_meas_geom(self, straight_line: bool) -> list[Tuple[float, float]]:
        """
        Adapt and modify the geometry of the soil measure and return the surface line as a list of points.
        The geometry is either extended to the hinterland as a straight line or it is extended in the direction of the
        dike's slope.

        Args:
            straight_line: if True, the geometry is a straight line from the BIT in the hinterland

        Returns:
            list of characteristic points for the dike's geometry

        """

        # EXT and BIT_0 are virtual points and must be removed
        self.geometry = self.geometry[self.geometry.index != "EXT"]
        self.geometry = self.geometry[self.geometry.index != "BIT_0"]

        # Continue the geometry in the hinterland as a straight line
        if straight_line:
            self.geometry.loc["BIT_1"] = [100, self.geometry.loc["BIT", "z"]]  # this is horizontal line to hinterland
            return [(x, z) for x, z in zip(self.geometry["x"].values, self.geometry["z"].values)]

        else:
            ##extend geometry LEFT SIDE
            start = (self.geometry.loc["BUK", "x"], self.geometry.loc["BUK", "z"])
            end = (self.geometry.loc["BUT", "x"], self.geometry.loc["BUT", "z"])
            x3, y3 = self.find_extended_end(start, end, 'left', 10)
            self.geometry.loc["left"] = [x3, y3]

            ##extend geometry RIGHT SIDE
            start = (self.geometry.loc["EBL", "x"], self.geometry.loc["EBL", "z"])
            end = (self.geometry.loc["BIT", "x"], self.geometry.loc["BIT", "z"])
            x4, y4 = self.find_extended_end(start, end, 'right', 10)
            self.geometry.loc["right"] = [x4, y4]

            # sort self.geometry by ascending x:
            self.geometry = self.geometry.sort_values(by="x")
            return [(x, z) for x, z in zip(self.geometry["x"].values, self.geometry["z"].values)]

    @staticmethod
    def find_extended_end(start_point: Tuple[float, float], end_point: Tuple[float, float], side: str,
                          desired_length: float) -> Tuple[float, float]:
        """
        Extend a line with a given length in a given direction.

        Args:
            start_point: start point of the line
            end_point: end point of the line
            side: if the line is extended on the right (hinterland) or left (foreland) of the embankment
            desired_length: length of the extension

        Returns:
            new end point of the extended line
        """
        sign = 1 if side == "right" else -1

        m = (end_point[1] - start_point[1]) / (end_point[0] - start_point[0])

        # Calculate the new endpoint of the extended line
        new_x = end_point[0] + sign * (desired_length / math.sqrt(1 + m ** 2))
        new_y = end_point[1] + m * (new_x - end_point[0])
        return new_x, new_y

    def find_polygons_to_fill_to_measure(self, polygons: list[Polygon]) -> list[Polygon]:
        """
        Find all the polygons that fill the gap between the original collection of polygons from a stix file and the new
        line of the top surface from the measure.

        Args:
            polygons: list of polygons from the initial situation (imported from a .stix file)

        Returns:
            The updated polygon list


        """
        # create two polygons that lie above and below the surface line to + infinity (=30) and - infinity (=-50)
        new_surface_line = LineString(self.measure_geometry_points)
        polygon_above_surface = Polygon([pt for pt in new_surface_line.coords] + [(100, 30), (-100, 30)])
        polygon_below_surface = Polygon([pt for pt in new_surface_line.coords] + [(100, -50), (-100, -50)])

        # merge all input polygons together
        merged_polygon = unary_union(polygons)

        # Do the difference between the merged polygon with the polygon lying above the surface line.
        # This returns an intermediate cropped polygon of the merged polygon that is below the surface line.
        cropped_polygon_below_surface = merged_polygon.difference(polygon_above_surface)

        # Now do the difference between the merged polygon with the polygon lying below the surface line.
        #  This returns one or several polygons.
        diff_poly_and_surface = polygon_below_surface.difference(merged_polygon)

        # Get the polygons that will fill the gap between the merged polygon and the surface line.
        filling_polygons = self.get_filling_polygons(diff_poly_and_surface, cropped_polygon_below_surface)

        return filling_polygons

    def get_filling_polygons(self, diff_poly_and_surface: Union[MultiPolygon, Polygon],
                             cropped_polygon_below_surface: Polygon) -> list[Polygon]:
        """
        Get the polygons that need to be filled to the top surface.

        Args:
            diff_poly_and_surface: The difference between the merged polygon and the polygon lying below the surface line
            cropped_polygon_below_surface: The cropped polygon that lies below the top surface line

        Returns:
            A list of all the polygons filling the gap between the merged polygon (the initial geometry) and the new
            surface line.

        """
        keep_poly = []

        # Find the intersection between the bounding box and the polygon
        bbox = self.get_bounding_box(self.measure_geometry_points, cropped_polygon_below_surface)
        intersection = diff_poly_and_surface.intersection(bbox)
        if isinstance(intersection, Polygon):  # if the intersection is a polygon, add it to the list directly
            keep_poly.append(intersection)
        else:
            for intersected_polygon in intersection.geoms:
                if isinstance(intersected_polygon, Polygon):  # eliminate weird linestrings
                    keep_poly.append(intersected_polygon)
        return keep_poly

    @staticmethod
    def get_bounding_box(top_surface: np.array, cropped_polygon_below_surface: Polygon) -> Polygon:
        """
        Get the bounding box of the polygon that is below the top surface line

        Args:
            top_surface: The geometry of measure's surface line
            cropped_polygon_below_surface: The cropped polygon that lies below the top surface line

        Returns:
            The bounding box of the polygon that is below the top surface line
        """
        min_x, min_y, max_x, _ = cropped_polygon_below_surface.bounds
        max_y = max(
            [p[1] for p in top_surface])  # the max of the bounding box should be the top of the measure geometry
        bounding_box = Polygon([(min_x, min_y), (min_x, max_y), (max_x, max_y), (max_x, min_y)])

        return bounding_box

    def apply_measure_to_dstability_model(self, fill_polygons: list[Polygon], path_intermediate_stix: Path) -> Path:
        """
        Apply the measure to the dstability model and save it to a new stix file.

        Args:
            fill_polygons: The polygons that are filling the gap between the initial geometry and the new surface line
            path_intermediate_stix: The path where the intermediate stix files are saved

        Returns:
            The Path of the new stix file
        """

        for stage_id in self._dstability_wrapper.get_all_stage_ids():
            self.modify_geometry(
                fill_polygons=fill_polygons,
                stage_id=stage_id)
        _original_name = self._dstability_wrapper.stix_path.stem
        new_file_name = self._dstability_wrapper.stix_path.with_stem(_original_name + f"_dberm_{self.dberm}_dcrest_{self.dcrest}").name

        self._dstability_wrapper.save_dstability_model(path_intermediate_stix / new_file_name)

        return new_file_name

    def modify_geometry(self, fill_polygons: list[Polygon], stage_id: int):
        """
        Modify in place the geometry of the dstability model for a given stage adding the filling polygons.
        The excess parts of the initial geometry above the surface line are NOT removed.
        The dike polygon and the filling polygons are NOT merged.

        Args:
            fill_polygons: The polygons that are filling the gap between the initial geometry and the new surface line
            stage_id (int): Id if the stage for which the geonetry is modified.

        """

        _layers = self._dstability_wrapper._dstability_model.datastructure.geometries[stage_id].Layers

        # 1. Loop over all other layers and modify in-place their geometry
        list_all_initial_point = []  # list of all the points in the initial stix
        for layer in _layers:
            poly = Polygon([(round(p.X, 3), round(p.Z, 3)) for p in layer.Points])
            self.modify_polygon_from_stix(current_polygon=poly,
                                          geometry_line=self.measure_geometry_points,
                                          layer=layer,
                                          list_all_point=list_all_initial_point)

        list_all_initial_point = list(set([tuple(p) for p in list_all_initial_point]))

        # 2. add the filling polygons and readjuste their coordinates when necessary.
        for fill_polygon in fill_polygons:

            if fill_polygon.area < 1:  # drop poylgons that are too small (below 1m2)
                continue

            # Consistency check: Fix the coordinates of the filling polygons
            list_points_to_add = self.fix_coordinates_filling_polygons(fill_polygon, list_all_initial_point)

            # Consistency check: remove duplicate point from list:
            new_list = []
            previous_point = GeolibPoint(x=999, z=999)
            for p in list_points_to_add:
                if p.x == previous_point.x and p.z == previous_point.z:
                    previous_point = p
                    continue
                else:
                    new_list.append(p)
                    previous_point = p

            # add layer to the model, keep the custom function until GEOLIB is updated/debugged
            self._dstability_wrapper._dstability_model.add_layer(
                points=new_list,
                soil_code="Dijksmateriaal", stage_id=stage_id)

    def modify_polygon_from_stix(self, current_polygon: Polygon, geometry_line: list[Tuple[float, float]],
                                 layer: PersistableLayer,
                                 list_all_point: Optional[list]):
        """
        Routine to make sure that the modified geometry is valid for D-Stability. It removes duplicate points that are
        too close and add intersections between polygons and the surface line.


        Args:
            current_polygon: The polygon that is modified
            geometry_line: The geometry of the surface line
            layer: The dstability layer object for which the collection of points is modified
            list_all_point: The list of all the points in the initial stix

        Returns:
            None
        """
        # this creates a new Polygon for the dike that incorporates the intersection points with the surface line.
        union_surface_dike = self.add_points_from_surface_intersection_to_polygon(initial_polygon=current_polygon,
                                                                                  geometry_line=geometry_line)
        for obj in union_surface_dike.geoms:
            if isinstance(obj, Polygon):
                list_points = [PersistablePoint(X=round(p[0], 3), Z=round(p[1], 3)) for p in obj.exterior.coords]
                list_points.pop()
                list_points = [list_points[-1]] + list_points
                list_points.pop()

                new_list = []
                previous_point = PersistablePoint(X=999, Z=999)
                # remove duplicate point from list:
                for p in list_points:
                    # if p.X == previous_point.X and p.Z == previous_point.Z:
                    if abs(p.X - previous_point.X) < 0.01 and abs(
                            p.Z - previous_point.Z) < 0.01:  # This condition is suppose to eliminate points from the dike polygon that are not supposed to exsit, i.e. the points added from the previous unary_union with the new geometry
                        previous_point = p
                        continue
                    else:
                        new_list.append(p)
                        previous_point = p

                # dike_layer.Points = list_points
                layer.Points = new_list
                list_all_point.extend([[p.X, p.Z] for p in new_list])

    def fix_coordinates_filling_polygons(self, fill_polygon: Polygon, list_all_initial_point: list) -> list[
        GeolibPoint]:
        """
        The coordinates of the intersection points between the fill_polygon and the polygon of the original geometry do not
        necessarily match because of very tiny inaccuracies in order of magnitude 1e-3. This leads to an inconsistent
        geometry in DStability.
        This routine loops over all the points of the fill_polygon, if it is close (tolerance 0.01) enough to a point
        already existing in the list_all_initial_point, then the point of the fill polygon is overwritten by the closest
        point from list_all_initial_point.

        Args:
            fill_polygon: One of the polygons that are filling the gap between the initial geometry and the new surface
            line.
            list_all_initial_point: List of all the points in the initial stix
        """

        list_points_to_add = []
        list_geolibpoint_filling = [GeolibPoint(x=round(pp[0], 3), z=round(pp[1], 3), tolerance=0.01) for pp in
                                    fill_polygon.exterior.coords]
        for i, geolib_pp in enumerate(list_geolibpoint_filling):

            a = False
            for initial_pt in list_all_initial_point:
                geolib_ppp = GeolibPoint(x=initial_pt[0], z=initial_pt[1], tolerance=0.01)

                # if geolib_ppp.__eq__(geolib_pp):
                if abs(geolib_ppp.z - geolib_pp.z) < 0.01 and abs(geolib_ppp.x - geolib_pp.x) < 0.01:
                    a = True
                    list_points_to_add.append(geolib_ppp)
                    if geolib_pp.x == -0:  # Consistency check: Eliminate the point with x=-0 which leads to errors
                        continue
                    break

            if not a:
                if geolib_pp.x == -0:  # Consistency check: Eliminate the point with x=-0 which leads to errors
                    continue
                list_points_to_add.append(geolib_pp)
        list_points_to_add.pop()
        return list_points_to_add

    def add_points_from_surface_intersection_to_polygon(self, initial_polygon: Polygon,
                                                        geometry_line: list[Tuple[float, float]]) -> \
            Union[MultiPolygon, Polygon]:
        """
        Add points from the intersection of the surface line and the initial polygon to the initial polygon.
        This function is mandatory in order to create a consistent geometry in the DStability model.


        Args:
            initial_polygon: initial polygon to which the points will be added
            geometry_line: geometry line of the surface level intersecting the initial polygon
        """

        ls_surface = LineString(geometry_line)

        # Modify in-place the dike polygon
        union_surface_dike = unary_union([ls_surface, initial_polygon])
        return union_surface_dike
