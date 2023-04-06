import math
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
import shapely
from geolib import DStabilityModel
from geolib.geometry import Point as GeolibPoint, Point
from geolib.models.dsettlement.internal import Layers
from geolib.models.dstability.analysis import DStabilityUpliftVanParticleSwarmAnalysisMethod, DStabilitySearchArea
from geolib.models.dstability.internal import PersistablePoint, PersistableLayer
from pandas import DataFrame
from shapely import Polygon, MultiPolygon, unary_union, LineString
import plotly.graph_objects as go
from shapely.ops import cascaded_union

from vrtool.stability_geometry_helper import crop_polygons_below_surface_line


def preprocess_dstab_model(dstability_model: DStabilityModel):
    """
    Preprocess the dstab model to make it more suitable for the VRTool
    """
    calc_setting = dstability_model.datastructure.calculationsettings[-1]
    print(calc_setting.UpliftVanParticleSwarm)

    i = calc_setting.UpliftVanParticleSwarm
    analysis_model = DStabilityUpliftVanParticleSwarmAnalysisMethod(
        search_area_a=DStabilitySearchArea(
            height=i.SearchAreaA.Height,
            top_left=Point(x=i.SearchAreaA.TopLeft.X,
                           z=i.SearchAreaA.TopLeft.Z),
            width=i.SearchAreaA.Width
        ),
        search_area_b=DStabilitySearchArea(
            height=i.SearchAreaB.Height,
            top_left=Point(x=i.SearchAreaB.TopLeft.X,
                           z=i.SearchAreaB.TopLeft.Z),
            width=i.SearchAreaB.Width),

        tangent_area_height=i.TangentArea.Height,
        tangent_area_top_z=i.TangentArea.TopZ,

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

    dstability_model.set_model(analysis_model, stage_id=0)
    dstability_model.meta.console_folder = Path(
        "C:\Program Files (x86)\Deltares\D-GEO Suite\D-Stability 2022.01.2/bin")

def modify_stix(stix_path: Path, fill_polygons: List[Polygon], measure_name: str, collection_polygon: List[Polygon], list_points_geom_me: List[Point]):
    _stix_path = Path(
        # r"C:\Users\hauth\OneDrive - Stichting Deltares\Documents\tempo/RW001.+096_STBI_maatgevend_Segment_38005_1D1.stix")
        r"C:\Users\hauth\OneDrive - Stichting Deltares\Desktop\projects\VRTools\TestCases/TestCase1_38-1_no_housing_SMALL/Stix/RW001.+096_STBI_maatgevend_Segment_38005_1D1.stix")
    _dstability_model = DStabilityModel()
    _dstability_model.parse(stix_path)

    preprocess_dstab_model(_dstability_model)




    # add_filling_polygons_to_dstability_model(_dstability_model, fill_polygons)
    stage_ids = [stage.Id for stage in _dstability_model.stages]

    for stage_id in stage_ids:
        # update_dstability_layers(
        #     dstability_model=_dstability_model,
        #
        #     # modified_polygons=new_polygons,
        #     modified_polygons=collection_polygon,
        #     stage_id=stage_id,
        #     merge_dijkmaterial=True,
        #     fill_polygons=fill_polygons
        # )

        test_sandbox(
            dstability_model=_dstability_model,

            modified_polygons=fill_polygons,
            # modified_polygons=collection_polygon,
            stage_id=stage_id,
            merge_dijkmaterial=False,
            fill_polygons=fill_polygons,
            list_points_geom_me=list_points_geom_me,
            crop=True

        )

    _output_folder = Path(r"C:\Users\hauth\OneDrive - Stichting Deltares\Documents\tempo")
    output_stix_name = f"{str(stix_path.parts[-1])[:-5]}_{measure_name}.stix"
    print(_output_folder.joinpath(output_stix_name))
    _dstability_model.serialize(_output_folder.joinpath(output_stix_name))
    # _dstability_model.execute()



def add_filling_polygons_to_dstability_model(dstability_model: DStabilityModel, fill_polygons: List[Polygon],
                                             stage_id: int):
    """
    Add the filling polygons to the dstability model for the given stage
    :param dstability_model: The dstability model to add the filling polygons to
    :param fill_polygons: The filling polygons to add
    :param stage_id: The stage id in the dstability model
    """

    # find layer with highest point:

    highest_point = 0
    highest_point_layer = None
    for layer in dstability_model.datastructure.geometries[stage_id].Layers:
        for point in layer.Points:
            if point.Z > highest_point:
                highest_point = point.Z
                highest_point_layer = layer

    # make a polygon from the highest point layer
    highest_point_layer_poly = Polygon([(p.X, p.Z) for p in highest_point_layer.Points])

    merge = unary_union(fill_polygons + [highest_point_layer_poly])

    add_layer_custom(model=dstability_model,
                     points=[GeolibPoint(x=p[0], z=p[1]) for p in merge.exterior.coords],
                     soil_code="Dijksmateriaal",
                     stage_id=stage_id
                     )
    return highest_point_layer_poly


def add_modified_polygon(dstability_model: DStabilityModel, polygons: List[Polygon]):
    new_stage_id = dstability_model.add_stage(label='new', notes='new')
    for poly in polygons:
        add_layer_custom(model=dstability_model,
                         points=[GeolibPoint(x=p[0], z=p[1]) for p in poly.exterior.coords],
                         soil_code="Dijksmateriaal", stage_id=new_stage_id)


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

def find_layer_id_dijk_layer(layers: List[PersistableLayer]) -> str:
    """
    Loop over all the layers from a .stix geometry and find the layer of the dike (assumed with the highest point) adn
    return its id.


    """
    highest_point = 0
    highest_point_layer = None
    for layer in layers:
        for point in layer.Points:
            if point.Z > highest_point:
                highest_point = point.Z
                highest_point_layer = layer
    return highest_point_layer.Id
def update_dstability_layers_with_cropped(dstability_model: DStabilityModel, modified_polygons: List[Polygon], stage_id: str,
                             merge_dijkmaterial: bool = True, fill_polygons: Optional[List[Polygon]] = None):
    geometry = dstability_model.datastructure.geometries[int(stage_id)]
    layers = geometry.Layers
    for layer, new_poly in zip(layers, modified_polygons):
        layer.Points = [PersistablePoint(X=p[0], Z=p[1]) for p in
                        new_poly.exterior.coords]  # this is a tricky case ... another layer has to be created

def update_dstability_layers(dstability_model: DStabilityModel, modified_polygons: List[Polygon], stage_id: str,
                             merge_dijkmaterial: bool = True, fill_polygons: Optional[List[Polygon]] = None):
    geometry = dstability_model.datastructure.geometries[int(stage_id)]
    layers = geometry.Layers

    if merge_dijkmaterial:
        id_dijk = find_layer_id_dijk_layer(layers)

    for layer, new_poly in zip(layers, modified_polygons):

        if merge_dijkmaterial:
            if id_dijk == layer.Id:
                highest_point_layer_poly = Polygon([(p.X, p.Z) for p in layer.Points])

                merge = unary_union(fill_polygons + [highest_point_layer_poly])
                if isinstance(merge, Polygon):
                    pp = [[p[0], p[1]] for p in merge.exterior.coords]

                    layer.Points = [PersistablePoint(X=p[0], Z=p[1]) for p in merge.exterior.coords]
                elif isinstance(merge, MultiPolygon):
                    print(1)
                    #find biggest polygon
                    # area = 0
                    # for merged_poly in merge.geoms:
                    #     if merged_poly.area > area:
                    #         area = merged_poly.area
                    #         biggest_poly = merged_poly

                    # layer.Points = [PersistablePoint(X=p[0], Z=p[1]) for p in biggest_poly.exterior.coords] # this is a tricky case ... another layer has to be created
                    layer.Points = [PersistablePoint(X=p[0], Z=p[1]) for p in merge.geoms[0].exterior.coords] # this is a tricky case ... another layer has to be created

        else:
            dstability_model.add_layer(points=[GeolibPoint(x=p[0], z=p[1]) for p in new_poly.exterior.coords],
                                       soil_code="Dijksmateriaal")


def test_sandbox(dstability_model: DStabilityModel, modified_polygons: List[Polygon], stage_id: str,
                             merge_dijkmaterial: bool = True, fill_polygons: Optional[List[Polygon]] = None, list_points_geom_me=None, crop: bool = False):

    geometry = dstability_model.datastructure.geometries[int(stage_id)]
    layers = geometry.Layers
    id_dijk = find_layer_id_dijk_layer(layers)
    dike_layer = [layer for layer in layers  if layer.Id == id_dijk][0]

    dike_polygon = Polygon([(p.X, p.Z) for p in dike_layer.Points])
    # if crop:
    # dike_polygon = crop_polygons_below_surface_line([dike_polygon], list_points_geom_me)[0]
    ls_surface = LineString(list_points_geom_me)

    union_surface_dike = unary_union([ls_surface, dike_polygon]) # this creates a new Polygon for the dike that incorporates
    # the intersection points with the surface line.

    for obj in union_surface_dike.geoms:
        if isinstance(obj, Polygon):
            list_points = [PersistablePoint(X=round(p[0], 3), Z=round(p[1], 3)) for p in
                                 obj.exterior.coords]
            list_points.pop()
            list_points = [list_points[-1]] + list_points
            list_points.pop()

            new_list = []
            previous_point = PersistablePoint(X=999, Z=999)
            # remove duplicate point from list:
            for p in list_points:
                if p.X == previous_point.X and p.Z == previous_point.Z:
                    previous_point = p
                    continue
                else:
                    new_list.append(p)
                    previous_point = p



            # dike_layer.Points = list_points
            dike_layer.Points = new_list



    # add the filling polygons and readjuste their coordinates when necessary.
    for fill_polygon in fill_polygons:

        if fill_polygon.area < 1: # drop poylgon that are too small (below 1m2)
            continue
        # if centroid is before 0
        if fill_polygon.centroid.coords[0][0] < 0:
            continue
        #
        liste = []
        list_geolibpoint_filling = [GeolibPoint(x=round(pp[0], 3), z=round(pp[1], 3), tolerance=0.01) for pp in fill_polygon.exterior.coords]
        for i, geolib_pp in enumerate(list_geolibpoint_filling):

            a = False
            for ppp in list_points:
                geolib_ppp = GeolibPoint(x=ppp.X, z=ppp.Z, tolerance=0.01)

                # a = geolib_ppp.__eq__(geolib_pp)
                # print(a, geolib_ppp, geolib_pp)
                if abs(geolib_ppp.z -geolib_pp.z) < 0.01 and abs(geolib_ppp.x - geolib_pp.x) < 0.01:
                    a = True
                    liste.append(geolib_ppp)
                    if geolib_pp.x == -0:
                        continue
                        print('geolib_pp YEYSYSYYSYS', geolib_pp)
                    break

            if not a:
                if geolib_pp.x == -0:
                    print('geolib_pp YEYSYSYYSYS', geolib_pp)
                liste.append(geolib_pp)
        liste.pop()

        new_list = []
        previous_point = GeolibPoint(x=999, z=999)
        # remove duplicate point from list:
        for p in liste:
            if p.x == previous_point.x and p.z == previous_point.z:
                previous_point = p
                continue
            else:
                new_list.append(p)
                previous_point = p

        add_layer_custom(model=dstability_model,
                             points=new_list,
                             soil_code="Dijksmateriaal", stage_id=int(stage_id))

    # Geometry consistency check: round all points to 3 decimals
    for layer in layers:
        if layer.Id == id_dijk:
            continue

        layer.Points = [PersistablePoint(X=round(p.X, 3), Z=round(p.Z, 3)) for p in
                             layer.Points]


def test_sandbox_merge(dstability_model: DStabilityModel, modified_polygons: List[Polygon], stage_id: str,
                             merge_dijkmaterial: bool = True, fill_polygons: Optional[List[Polygon]] = None, list_points_geom_me=None, crop: bool = False):

    geometry = dstability_model.datastructure.geometries[int(stage_id)]
    layers = geometry.Layers
    id_dijk = find_layer_id_dijk_layer(layers)
    dike_layer = [layer for layer in layers  if layer.Id == id_dijk][0]

    dike_polygon = Polygon([(p.X, p.Z) for p in dike_layer.Points])
    # if crop:
    # dike_polygon = crop_polygons_below_surface_line([dike_polygon], list_points_geom_me)[0]


    # merge adjacent polygons to dike_polygons:
    # for fill_poly in fill_polygons:
    #     if dike_polygon.touches(fill_poly):
    #         dike_polygon = unary_union([dike_polygon, fill_poly])

    dike_polygon = unary_union([fill_polygons + [dike_polygon]])

    if isinstance(dike_polygon, Polygon):
        list_points = [PersistablePoint(X=p[0], Z=p[1]) for p in
                             dike_polygon.exterior.coords]

        list_points.pop()
        list_points = [list_points[-1]] + list_points
        dike_layer.Points = list_points    # this is a tricky case ... another layer has to be created
    elif isinstance(dike_polygon, MultiPolygon):
        area = 0
        for merged_poly in dike_polygon.geoms:
            if merged_poly.area > area:
                area = merged_poly.area
                biggest_poly = merged_poly

        list_points = [PersistablePoint(X=p[0], Z=p[1]) for p in
                       biggest_poly.exterior.coords]

        list_points.pop()
        list_points = [list_points[-1]] + list_points
        dike_layer.Points = list_points  # this is a tricky case ... another layer has to be created




def rebuild_geom(dstability_model: DStabilityModel, modified_polygons: List[Polygon], stage_id: str,
                             merge_dijkmaterial: bool = True, fill_polygons: Optional[List[Polygon]] = None):

    stage_id = dstability_model.add_stage('NEW', 'NEW')
    # geometry = dstability_model.datastructure.geometries[int(stage_id)]
    # layers = geometry.Layers

    for poly in modified_polygons:
        # add_layer_custom(model=dstability_model,
        #                  points=[GeolibPoint(x=p[0], z=p[1]) for p in poly.exterior.coords],
        #                  soil_code="Dijksmateriaal", stage_id=stage_id)
        dstability_model.add_layer(points=[GeolibPoint(x=p[0], z=p[1]) for p in poly.exterior.coords],
                         soil_code="Dijksmateriaal", stage_id=stage_id)



def plot_polygon(polygon_list: List[Polygon], surface_line: np.array):
    """
    Plot the polygon list
    :param polygon_list: The polygon list to plot
    :return: None
    """
    fig = go.Figure()

    for i, poly in enumerate(polygon_list):
        fig.add_trace(go.Scatter(
            x=[p[0] for p in poly.exterior.coords],
            y=[p[1] for p in poly.exterior.coords],
            mode="lines",
            fill="toself",
            name="Polygon {}".format(i)))

    fig.add_trace(go.Scatter(x=[p[0] for p in surface_line],
                             y=[p[1] for p in surface_line],
                             mode="lines",
                             line=dict(color="black", width=5),
                             ))

    fig.show()


def get_boundary_box_from_polygons(polygons: List[Polygon]) -> Polygon:
    """
    Get the boundary box of the polygon list
    :param polygons: The polygon list
    :return: The boundary box
    """
    # get the boundary box
    x_min = min([p.bounds[0] for p in polygons])
    y_min = min([p.bounds[1] for p in polygons])
    x_max = max([p.bounds[2] for p in polygons])
    y_max = max([p.bounds[3] for p in polygons])

    boundary_box = Polygon([(x_min, y_min), (x_min, y_max), (x_max, y_max), (x_max, y_min)])

    return boundary_box


def redefine_top_surface_polygons(polygons: List[Polygon], top_surface: np.array) -> List[Polygon]:
    """
    Redefine the polygon from clustering to match with the top surface.

    :param polygon_list: The current clustered polygon list
    :param top_surface: The top surface that should be included to the top polygons
    :return: The updated polygon list
    """

    points_ahn = [shapely.Point(i) for i in top_surface]

    new_line = LineString(points_ahn)
    ahn_polygon_top = Polygon([pt for pt in new_line.coords] + [(100, 30), (-100, 30)])
    ahn_polygon_bot = Polygon([pt for pt in new_line.coords] + [(100, -30), (-100, -30)])

    list_modified_polygons = []

    # fig = go.Figure()
    #
    # fig.add_trace(go.Scatter(x=[p[0] for p in top_surface],
    #                          y=[p[1] for p in top_surface],
    #                          mode='lines',
    #                          line=dict(color='black', width=5),
    #                          name=f'meas'
    #                          )
    #               )
    # for poly in new_polygons:
    # for poly in polygons:
    #     fig.add_trace(go.Scatter(
    #         name='Base_polygon',
    #         x=[p[0] for p in poly.exterior.coords],
    #         y=[p[1] for p in poly.exterior.coords],
    #         mode='lines',
    #         line=dict(color='grey'),
    #         fill='toself',
    #         fillcolor='grey',
    #         opacity=0.2,
    #         legendgroup=True
    #     ))
    # merge all polygons to one polygon
    merged_polygon = [unary_union(polygons)]
    for i, poly in enumerate(merged_polygon):
        # if i== 1:
        # fig.add_trace(go.Scatter(
        #     name='Modified_polygon',
        #     x=[p[0] for p in poly.exterior.coords],
        #     y=[p[1] for p in poly.exterior.coords],
        #     mode='lines',
        #     line=dict(color='red'),
        #     fill='toself',
        # ))
        geometric_polygon = poly

        if not geometric_polygon.intersects(
                new_line):  # if polygon does not intersect with the top surface, keep it as it is
            list_modified_polygons.append(poly)
            continue

        min_poly_y = min([pt[1] for pt in geometric_polygon.exterior.coords])

        # Do the difference between the polygon from clustering with the polygon lying above the AHN surface to
        # + infinity. This returns an intermediate cropped polygon of the clustering polygon that is below the AHN surface
        cropped_polygon_below_ahn = poly.difference(ahn_polygon_top)

        # Now do the difference between the polygon from clustering with the polygon lying below the AHN surface to
        # - infinity. This returns one or several polygons that need filtering to be kept.
        diff_poly_and_ahn = ahn_polygon_bot.difference(poly)

        # if i== 0:
        #     # fig.add_trace(go.Scatter(
        #     #     name='ahn_top',
        #     #     x=[p[0] for p in ahn_polygon_top.exterior.coords],
        #     #     y=[p[1] for p in ahn_polygon_top.exterior.coords],
        #     #     mode='lines',
        #     #     line=dict(color='red'),
        #     #     fill='toself',
        #     #     opacity=0.5,
        #     # ))
        #     fig.add_trace(go.Scatter(
        #         name='ahn_bot',
        #         x=[p[0] for p in ahn_polygon_bot.exterior.coords],
        #         y=[p[1] for p in ahn_polygon_bot.exterior.coords],
        #         mode='lines',
        #         line=dict(color='red'),
        #         fill='toself',
        #         opacity=0.5,
        #     ))
        #
        #     fig.add_trace(go.Scatter(
        #         name='dif_top',
        #         x=[p[0] for p in cropped_polygon_below_ahn.exterior.coords],
        #         y=[p[1] for p in cropped_polygon_below_ahn.exterior.coords],
        #         mode='lines',
        #         line=dict(color='blue'),
        #         fill='toself',
        #     ))
        #
        #     # for poly in diff_poly_and_ahn.geoms:
        #     #     fig.add_trace(go.Scatter(
        #     #         name='bbox',
        #     #         x=[p[0] for p in boundary_box.exterior.coords],
        #     #         y=[p[1] for p in boundary_box.exterior.coords],
        #     #         mode='lines',
        #     #         line=dict(color='green'),
        #     #         fill='toself',
        #     #     ))
        #
        #
        # #
        keep_poly = []

        def merge_all_possible_polygon(diff_poly_and_ahn_0: MultiPolygon, diff_top, keep_poly):
            # find bounding box of the polygon
            min_x, min_y, max_x, max_y = diff_top.bounds

            # max_x = 80
            # Make a rectangle of the bounding box:
            bounding_box = Polygon([(min_x, min_y), (min_x, max_y), (max_x, max_y), (max_x, min_y)])
            # Find the intersection between the bounding box and the polygon

            intersection = diff_poly_and_ahn_0.intersection(bounding_box)

            # print(intersection)
            for poly in intersection.geoms:
                if isinstance(poly, Polygon):
                    keep_poly.append(poly)
                    # fig.add_trace(go.Scatter(
                    #     name='diff_bot',
                    #     x=[p[0] for p in poly.exterior.coords],
                    #     y=[p[1] for p in poly.exterior.coords],
                    #     mode='lines',
                    #     line=dict(color='green'),
                    #     fill='toself',
                    # ))

        # merge_all_possible_polygon(diff_poly_and_ahn, cropped_polygon_below_ahn, keep_poly)

        if isinstance(diff_poly_and_ahn, MultiPolygon):

            merge_all_possible_polygon(diff_poly_and_ahn, cropped_polygon_below_ahn, keep_poly)
            # for pol in diff_poly_and_ahn.geoms:
            #     min_poly_y_it = min([pt[1] for pt in pol.exterior.coords])
            #     if min_poly_y_it > min_poly_y:  # only keep sub-polygons that are higher than the clustering polygon
            #         keep_poly.append(pol)

        elif isinstance(diff_poly_and_ahn, Polygon):
            min_poly_y_it = min([pt[1] for pt in diff_poly_and_ahn.exterior.coords])
            if min_poly_y_it > min_poly_y:
                keep_poly.append(diff_poly_and_ahn)

        final_poly = unary_union(keep_poly)
        if isinstance(final_poly, Polygon):
            list_modified_polygons.append(final_poly)
        elif isinstance(final_poly, MultiPolygon):
            for subpoly in final_poly.geoms:
                list_modified_polygons.append(subpoly)
        if isinstance(final_poly, MultiPolygon):
            NotImplementedError(
                "Check why the modified polygon is a multipolygon, possible explanation: surface line is not elongated enough")

    # Checking loop
    for i, poly_i in enumerate(list_modified_polygons):
        for j, poly_j in enumerate(list_modified_polygons):
            if i == j:
                continue
            if poly_i.overlaps(poly_j):
                min_y_poly_i = min([pt[1] for pt in poly_i.exterior.coords])
                min_y_poly_j = min([pt[1] for pt in poly_j.exterior.coords])
                if min_y_poly_i > min_y_poly_j:
                    continue
                else:
                    diff = poly_i.difference(poly_j)
                    biggest_poly = None
                    area = 0

                    # The difference creates a lot of irrelevant very small polygons that must be filtered out.
                    if isinstance(diff, Polygon):
                        list_modified_polygons[i] = diff
                    elif isinstance(diff, MultiPolygon):
                        for pol in diff.geoms:
                            if pol.area > area:
                                biggest_poly = pol
                                area = pol.area

                        list_modified_polygons[i] = biggest_poly
                        poly_i = biggest_poly  # necessary as the polygon is modified in-place in the list

    # for poly in list_modified_polygons:
    #     fig.add_trace(go.Scatter(
    #         name='layer',
    #         x=[p[0] for p in poly.exterior.coords],
    #         y=[p[1] for p in poly.exterior.coords],
    #         mode='lines',
    #         fill='toself', ))
    # fig.show()
    return list_modified_polygons


# modify_stix(None, None)


def plot_measure_profile(polygons: Dict[str, List[Polygon]], surface):
    fig = go.Figure()
    #
    # dict(inital_polygons=collection_polygon,
    #      fill_polygons=fill_polygons,
    #      new_polygons=new_polygons,
    #      )

    fig.add_trace(go.Scatter(x=[p[0] for p in surface],
                             y=[p[1] for p in surface],
                             mode='lines',
                             line=dict(color='black', width=5),
                             name=f'Measure surface line'
                             )
                  )
    for i, poly in enumerate(polygons['initial_polygons']):
        fig.add_trace(go.Scatter(
            name='intitial_polygons',
            x=[p[0] for p in poly.exterior.coords],
            y=[p[1] for p in poly.exterior.coords],
            mode='lines',
            fillcolor='grey',
            opacity=0.3,
            line=dict(color='grey'),
            showlegend=True if i == 0 else False,
            legendgroup='initial_polygons',
            fill='toself', ))

    for i, poly in enumerate(polygons['fill_polygons']):
        fig.add_trace(go.Scatter(
            name='filling',
            x=[p[0] for p in poly.exterior.coords],
            y=[p[1] for p in poly.exterior.coords],
            mode='lines',
            fillcolor='green',
            opacity=0.9,
            line=dict(color='green'),
            showlegend=True if i == 0 else False,
            legendgroup='filling',
            fill='toself', ))

    for i, poly in enumerate(polygons['new_polygons']):
        fig.add_trace(go.Scatter(
            name='new_polygons',
            x=[p[0] for p in poly.exterior.coords],
            y=[p[1] for p in poly.exterior.coords],
            mode='lines',
            fillcolor='red',
            opacity=0.7,
            line=dict(color='red'),
            showlegend=True if i == 0 else False,
            legendgroup='new_polygons',
            fill='toself', ))
    fig.show()
    return



 # for layer, new_poly in zip(layers, modified_polygons):
 #        if id_dijk == layer.Id:
 #            dike_polygon = Polygon([(p.X, p.Z) for p in layer.Points])
 #            print(dike_polygon.is_valid)
 #            intersection_points_profile = dike_polygon.intersection(ls_surface)
 #
 #
 #            list_of_all_points_to_be_added = []
 #            for line in intersection_points_profile.geoms:
 #                if isinstance(line, LineString):
 #                    list_of_all_points_to_be_added.append(line.coords[0])
 #                    list_of_all_points_to_be_added.append(line.coords[-1])
 #
 #            list_point = [(p.X, p.Z) for p in layer.Points] + list_of_all_points_to_be_added
 #            new_polygon = Polygon(list_point)
 #            print(new_polygon)
 #            print(new_polygon.is_valid)
 #            layer.Points = [PersistablePoint(X=p[0], Z=p[1]) for p in
 #                            new_polygon.geoms[0].exterior.coords]  # this is a tricky case ... another layer has to be created


def clockwiseangle_and_distance(point, origin=[6, 6], refvec=[0,1]):
    # Vector between point and the origin: v = p - o
    vector = [point[0]-origin[0], point[1]-origin[1]]
    # Length of vector: ||v||
    lenvector = math.hypot(vector[0], vector[1])
    # If length is zero there is no angle
    if lenvector == 0:
        return -math.pi, 0
    # Normalize vector: v/||v||
    normalized = [vector[0]/lenvector, vector[1]/lenvector]
    dotprod  = normalized[0]*refvec[0] + normalized[1]*refvec[1]     # x1*x2 + y1*y2
    diffprod = refvec[1]*normalized[0] - refvec[0]*normalized[1]     # x1*y2 - y1*x2
    angle = math.atan2(diffprod, dotprod)
    # Negative angles represent counter-clockwise angles so we need to subtract them
    # from 2*pi (360 degrees)
    if angle < 0:
        return 2*math.pi+angle, lenvector
    # I return first the angle because that's the primary sorting criterium
    # but if two vectors have the same angle then the shorter distance should come first.
    return angle, lenvector