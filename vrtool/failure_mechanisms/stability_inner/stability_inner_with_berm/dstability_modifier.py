from pathlib import Path
from typing import List, Tuple, Dict, Optional

from geolib import DStabilityModel
from geolib.geometry import Point as GeolibPoint
from geolib.models.dstability.analysis import DStabilityUpliftVanParticleSwarmAnalysisMethod, DStabilitySearchArea
from geolib.models.dstability.internal import PersistablePoint
from shapely import Polygon, LineString, unary_union


def apply_measure_to_dstab(stix_path: Path, measure_geometry_points: List[Tuple[float, float]], polygons_dict,
                           measure_name: str):
    """

    :param stix_path: The path to the original stix file
    :param measure_geometry_points: The list of points that define the geometry of the measure
    :param polygons_dict: The dictionary of polygons that define the geometry of the measure
    :param measure_name: name of the measure, only used for the path of the output stix file
    """

    # Make a new dstability model and parse the original stix for every measure.
    _dstability_model = DStabilityModel()
    _dstability_model.parse(stix_path)

    preprocess_dstab_model(_dstability_model)

    stage_ids = [stage.Id for stage in _dstability_model.stages]

    for stage_id in stage_ids:
        modify_geometry(dstability_model=_dstability_model,
                        polygons_dict=polygons_dict,
                        stage_id=stage_id,
                        measure_geometry_points=measure_geometry_points)

    # Save the modified stix file to a dedicated folder
    _output_folder = Path(r"C:\Users\hauth\OneDrive - Stichting Deltares\Documents\tempo")
    # _output_folder = Path(r"C:\Users\hauth\OneDrive - Stichting Deltares\Desktop\projects\VRTools\output")
    output_stix_name = f"{str(stix_path.parts[-1])[:-5]}_{measure_name}.stix"
    _dstability_model.serialize(_output_folder.joinpath(output_stix_name))
    # _dstability_model.execute() # execute command must be after serialize

def preprocess_dstab_model(dstability_model: DStabilityModel):
    """
    Preprocess the dstability model to make it sure that it is ready for the analysis
    """
    # For TestCase 38-1, the initial stage does not have calculations settings, so we copy them for the last stage
    calc_setting = dstability_model.datastructure.calculationsettings[-1] # fetch calc setting of last stage

    i = calc_setting.UpliftVanParticleSwarm
    analysis_model = DStabilityUpliftVanParticleSwarmAnalysisMethod(
        search_area_a=DStabilitySearchArea(
            height=i.SearchAreaA.Height,
            top_left=GeolibPoint(x=i.SearchAreaA.TopLeft.X,
                           z=i.SearchAreaA.TopLeft.Z),
            width=i.SearchAreaA.Width
        ),
        search_area_b=DStabilitySearchArea(
            height=i.SearchAreaB.Height,
            top_left=GeolibPoint(x=i.SearchAreaB.TopLeft.X,
                           z=i.SearchAreaB.TopLeft.Z),
            width=i.SearchAreaB.Width),

        tangent_area_height=i.TangentArea.Height,
        tangent_area_top_z=i.TangentArea.TopZ,

        # TODO add the constraints:

        # slip_plane_constraints=DStabilitySlipPlaneConstraints(
        #     is_size_constraints_enabled=True,
        #     is_zone_a_constraints_enabled=True,
        #     is_zone_b_constraints_enabled=True,
        #     minimum_slip_plane_depth=1,
        #     minimum_slip_plane_length=1,
        #     width_zone_a=8,
        #     width_zone_b=30,
        #     x_left_zone_a=start_zone_a,
        #     x_left_zone_b=0,
    )

    dstability_model.set_model(analysis_model, stage_id=0) # apply calc setting to initial stage
    dstability_model.meta.console_folder = Path(
        "C:\Program Files (x86)\Deltares\D-GEO Suite\D-Stability 2022.01.2/bin")  # This line is necessary to be able to
    # execute the model locally. The path has to be adapted to the user's computer.





def modify_geometry(dstability_model: DStabilityModel, polygons_dict: Dict[str, List[Polygon]], stage_id: str,

                 measure_geometry_points: List[Tuple[float, float]]):
    """
    Modify in place the geometry of the dstability model for a given stage adding the filling polygons.
    The excess parts of the initial geometry above the surface line are NOT removed.
    The dike polygon and the filling polygons are NOT merged.

    """
    _fill_polygons = polygons_dict['fill_polygons']

    _layers = dstability_model.datastructure.geometries[int(stage_id)].Layers

    # 1. Loop over all other layers and modify in-place their geometry
    list_all_initial_point = []  # list of all the points in the initial stix
    for layer in _layers:
        poly = Polygon([(round(p.X, 3), round(p.Z, 3)) for p in layer.Points])
        modify_polygon_from_stix(current_polygon=poly,
                                 geometry_line=measure_geometry_points,
                                 layer=layer,
                                 list_all_point=list_all_initial_point)

    list_all_initial_point = list(set([tuple(p) for p in list_all_initial_point]))

    # 2. add the filling polygons and readjuste their coordinates when necessary.
    for fill_polygon in _fill_polygons:

        if fill_polygon.area < 1:  # drop poylgons that are too small (below 1m2)
            continue

        # Consistency check: Fix the coordinates of the filling polygons
        list_points_to_add = fix_coordinates_filling_polygons(fill_polygon, list_all_initial_point)

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
        add_layer_custom(model=dstability_model,
                         points=new_list,
                         soil_code="Dijksmateriaal", stage_id=int(stage_id))

def modify_polygon_from_stix(current_polygon: Polygon, geometry_line, layer,
                             list_all_point: Optional[List]):
    union_surface_dike = add_points_from_surface_intersection_to_polygon(initial_polygon=current_polygon,
                                                   geometry_line=geometry_line,
                                                   )
    # this creates a new Polygon for the dike that incorporates the intersection points with the surface line.
    # TODO not so satisfied with this technique to get the intersection points, it sometimes to inaccuracies of the points

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

def fix_coordinates_filling_polygons(fill_polygon: Polygon, list_all_initial_point: List) -> List[GeolibPoint]:
    """
    The coordinates of the intersection points between the fill_polygon and the polygon of the original geometry do not
    necessarily match because of very tiny inaccuracies in order of magnitude 1e-3. This leads to an inconsistent
    geometry in DStability.
    This routine loops over all the points of the fill_polygon, if it is close (tolerance 0.01) enough to a point
    already existing in the list_all_initial_point, then the point of the fill polygon is overwritten by the closest
    point from list_all_initial_point.

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

def add_points_from_surface_intersection_to_polygon(initial_polygon, geometry_line=List[Tuple[float, float]]) -> Union[MultiPolygon, Polygon]:
    """
    Add points from the intersection of the surface line and the initial polygon to the initial polygon.
    This function is mandatory in order to create a consistent geometry in the DStability model.

    :param initial_polygon: initial polygon to which the points will be added
    :param geometry_line: geometry line of the surface level intersecting the initial polygon

    """

    ls_surface = LineString(geometry_line)

    # Modify in-place the dike polygon
    # TODO: find a better strategy that unary union, because it might lead to inacurracies at precision 1-3
    union_surface_dike = unary_union([ls_surface, initial_polygon])
    return union_surface_dike

def add_layer_custom(
        model: DStabilityModel,
        points: List[GeolibPoint],
        soil_code: str,
        label: str = "",
        notes: str = "",
        stage_id: int = None,
) -> int:
    """
    Add a soil layer to the model

    Args:
        points (List[Point]): list of Point classes, in clockwise order (non closed simple polygon)
        soil_code (str): code of the soil for this layer
        label (str): label defaults to empty string
        notes (str): notes defaults to empty string
        stage_id (int): stage to add to, defaults to 0

    Returns:
        int: id of the added layer
    """
    stage_id = stage_id if stage_id is not None else model.current_stage

    if not model.datastructure.has_stage(stage_id):
        raise IndexError(f"stage {stage_id} is not available")

    geometry = model.datastructure.geometries[stage_id]
    soillayerscollection = model.datastructure.soillayers[stage_id]

    # do we have this soil code?
    if not model.soils.has_soilcode(soil_code):
        raise ValueError(
            f"The soil with code {soil_code} is not defined in the soil collection."
        )

    # add the layer to the geometry
    # the checks on the validity of the points are done in the PersistableLayer class
    persistable_layer = geometry.add_layer(
        id=str(model._get_next_id()), label=label, points=points, notes=notes
    )

    # add the connection between the layer and the soil to soillayers
    soil = model.soils.get_soil(soil_code)
    soillayerscollection.add_soillayer(layer_id=persistable_layer.Id, soil_id=soil.id)
    return int(persistable_layer.Id)