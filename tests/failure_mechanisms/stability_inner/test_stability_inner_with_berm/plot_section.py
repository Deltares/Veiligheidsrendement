from pathlib import Path
from shutil import rmtree
from typing import Dict, List

from geolib import DStabilityModel
from shapely import Polygon
import plotly.graph_objects as go


from vrtool.decision_making.measures import SoilReinforcementMeasure
from vrtool.defaults.vrtool_config import VrtoolConfig
from vrtool.failure_mechanisms.stability_inner.stability_inner_with_berm.berm_polygon_maker import \
    get_modified_meas_geom, find_polygons_to_fill_to_measure, crop_polygons_below_surface_line
from vrtool.failure_mechanisms.stability_inner.stability_inner_with_berm.dstability_modifier import \
    apply_measure_to_dstab
from vrtool.flood_defence_system.dike_traject import DikeTraject
from vrtool.run_workflows.measures_workflow.run_measures import RunMeasures
from vrtool.run_workflows.vrtool_plot_mode import VrToolPlotMode




# ====== IMPORTANT ======== #
# The initial stix must be pre-processed before using them with the prototype. They need to be executed blankly first
# otherwise the serialization will fail.

MAPPING_STIX_DV = {"DV01A": "RW001.+096_STBI_maatgevend_Segment_38005_1D1.stix",
                   "DV01B": "RW002.+035_STBI_representatief_Segment_38005_1D1.stix",
                   "DV04": "RW018.+035_STBI_representatief_Segment_38005_1D1.stix",
                   "DV12": "RW062.+028_STBI_maatgevend_Segment_38004_A_1D2.stix",
                   "DV15": "RW077.+004_STBI_representatief_Segment_38004_A_1D2.stix",
                   "DV33A": "RW162.+075_STBI_representatief_Segment_38003_C_1D2b.stix",
                   "DV34": "RW169.+020_STBI_representatief_Segment_38003_C_1D2b.stix",
                   "DV37B": "RW190.+008_STBI_representatief_Segment_38002_1D8.stix",
                   "DV40": "RW201.+019_STBI_maatgevend_Segment_38002_1D8.stix",
                   "DV50": "RW258.+091_STBI_representatief_Segment_38001_A_1D2.stix",
                   }



# 1. Define input and output directories..
_vrtool_dir = Path(r"C:\Users\hauth\OneDrive - Stichting Deltares\Desktop\projects\VRTools\TestCases")
_input_model = _vrtool_dir / "TestCase1_38-1_no_housing_SMALL"
_results_dir = _vrtool_dir / "sandbox_results"
if _results_dir.exists():
    rmtree(_results_dir)

# 2. Define the configuration to use.
_vr_config = VrtoolConfig()
_vr_config.input_directory = _input_model
_vr_config.output_directory = _results_dir
_vr_config.traject = "38-1"
_plot_mode = VrToolPlotMode.STANDARD


# 3. Run the measures workflow.
_selected_traject = DikeTraject.from_vr_config(_vr_config)
_measures = RunMeasures(_vr_config, _selected_traject, plot_mode=_plot_mode)
_measures_result = _measures.run()


# stop

for section in _selected_traject.sections:
    print(f'============NEW SECTION {section.name} =============')

    stix_name = MAPPING_STIX_DV[section.name]
    stix_path = _input_model.joinpath("Stix/" + stix_name)
    _dstability_model = DStabilityModel()
    _dstability_model.parse(stix_path)

    # Get all the polygons from the stix file.
    # It is assumed that all the stages share the same surface line, so we only need to find the filling polygons for
    # the first stage. If False, then this routine must be rerun for every stage.
    collection_polygon = [Polygon([(p.X, p.Z) for p in layer.Points]) for layer in _dstability_model.datastructure.geometries[0].Layers]

    solution = _measures_result.solutions_dict[section.name]
    measures = solution.measures

    for meas in measures:
        if isinstance(meas, SoilReinforcementMeasure):
            for i, soil_measure in enumerate(meas.measures):
                if i != 10:
                    continue
                _measure_geometry_points = get_modified_meas_geom(soil_measure=soil_measure, straight_line=False)

                # 1. Run first routine to find the polygons to fill.
                _fill_polygons = find_polygons_to_fill_to_measure(collection_polygon, _measure_geometry_points)
                _new_polygons = crop_polygons_below_surface_line(collection_polygon, _measure_geometry_points)
                _polygon_final = _fill_polygons + _new_polygons
                _polygons_dict = dict(
                    initial_polygons=collection_polygon,
                    fill_polygons=_fill_polygons,
                    new_polygons=_new_polygons)

                # plot_measure_profile(_polygons_dict, _measure_geometry_points, title=f'{section.name}_berm_{i}')

                # 2. Run second routine to apply the measure to the dstability model.
                apply_measure_to_dstab(stix_path, _measure_geometry_points, _polygons_dict, str(i))




def plot_measure_profile(polygons: Dict[str, List[Polygon]], surface, title=str):
    fig = go.Figure()

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

    # for i, poly in enumerate(polygons['new_polygons']):
    #     fig.add_trace(go.Scatter(
    #         name='new_polygons',
    #         x=[p[0] for p in poly.exterior.coords],
    #         y=[p[1] for p in poly.exterior.coords],
    #         mode='lines',
    #         fillcolor='red',
    #         opacity=0.7,
    #         line=dict(color='red'),
    #         showlegend=True if i == 0 else False,
    #         legendgroup='new_polygons',
    #         fill='toself'))
    fig.update_layout(title=title)
    fig.show()
    return
