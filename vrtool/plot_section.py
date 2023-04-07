import pickle
from pathlib import Path
from shutil import rmtree
from typing import List

from geolib import DStabilityModel
from shapely import Polygon
import plotly.graph_objects as go

from vrtool.decision_making.measures import SoilReinforcementMeasure
from vrtool.defaults.vrtool_config import VrtoolConfig
from vrtool.flood_defence_system.dike_traject import DikeTraject
from vrtool.run_workflows.measures_workflow.results_measures import ResultsMeasures
from vrtool.run_workflows.measures_workflow.run_measures import RunMeasures
from vrtool.run_workflows.vrtool_plot_mode import VrToolPlotMode
from vrtool.stability_geometry_helper import find_polygons_to_fill_to_measure, crop_polygons_below_surface_line
from vrtool.try_surface_modifier import plot_measure_profile, modify_stix, get_modified_meas_geom

# path_to_stixfolder = Path(
#     r"C:\Users\hauth\OneDrive - Stichting Deltares\Desktop\projects\VRTools\TestCases\TestCase1_38-1_no_housing_SMALL\Stix")
#
# stix = path_to_stixfolder.joinpath("RW001.+096_STBI_maatgevend_Segment_38005_1D1.stix")

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

# stix_name = MAPPING_STIX_DV["DV01A"]
# stix_path1 = path_to_stixfolder.joinpath(stix_name)
#
# stix_path = Path(
#     # r"C:\Users\hauth\OneDrive - Stichting Deltares\Documents\tempo/RW001.+096_STBI_maatgevend_Segment_38005_1D1.stix")
# r"C:\Users\hauth\OneDrive - Stichting Deltares\Desktop\projects\VRTools\TestCases/TestCase1_38-1_no_housing_SMALL/Stix/RW001.+096_STBI_maatgevend_Segment_38005_1D1.stix")
#
# print(stix_path)
# print(stix_path1)
# print(stix_path == stix_path1)
# _dstability_model = DStabilityModel()
# _dstability_model.parse(stix_path)
#
# _dstability_model.meta.console_folder = Path(
#     "C:\Program Files (x86)\Deltares\D-GEO Suite\D-Stability 2022.01.2/bin")
# _dstability_model.serialize(Path(r"C:\Users\hauth\OneDrive - Stichting Deltares\Documents\tempo/temp11.stix"))
# _dstability_model.execute()
#
# stop

# 1. Define input and output directories..
_vrtool_dir = Path(r"C:\Users\hauth\OneDrive - Stichting Deltares\Desktop\projects\VRTools\TestCases")
_input_model = _vrtool_dir / "TestCase1_38-1_no_housing_SMALL"
# _input_model = _vrtool_dir / "integrated_SAFE_16-3_small_FULL"
assert _input_model.exists(), "No input model found at {}".format(_input_model)

_results_dir = _vrtool_dir / "sandbox_results"
if _results_dir.exists():
    rmtree(_results_dir)

# 2. Define the configuration to use.
_vr_config = VrtoolConfig()
_vr_config.input_directory = _input_model
_vr_config.output_directory = _results_dir
_vr_config.traject = "16-3"
_plot_mode = VrToolPlotMode.STANDARD

# 3. "Run" the model.
# Step 0. Load Traject
_selected_traject = DikeTraject.from_vr_config(_vr_config)

# # Step 1. Safety assessment.
# _safety_assessment = RunSafetyAssessment(
#     _vr_config, _selected_traject, plot_mode=_plot_mode
# )
# _safety_result = _safety_assessment.run()

print("DO measure")

# # Step 2. Measures.
_measures = RunMeasures(_vr_config, _selected_traject, plot_mode=_plot_mode)
_measures_result = _measures.run()


# stop

for section in _selected_traject.sections:
    print(f'============NEW SECTION {section.name} =============')

    stix_name = MAPPING_STIX_DV[section.name]
    stix_path = _input_model.joinpath("Stix/" + stix_name)
    _dstability_model = DStabilityModel()
    _dstability_model.parse(stix_path)

    # Get layer polygons
    collection_polygon = []
    for layer in _dstability_model.datastructure.geometries[0].Layers:
        points = layer.Points
        poly = Polygon([(p.X, p.Z) for p in points])
        collection_polygon.append(poly)

    solution = _measures_result.solutions_dict[section.name]

    measures = solution.measures

    for meas in measures:
        if isinstance(meas, SoilReinforcementMeasure):

            for i, soil_measure in enumerate(meas.measures):
                if i != 30:
                    continue
                # transform df in list of point:
                list_points_geom_me = get_modified_meas_geom(soil_measure=soil_measure,
                                                             straight_lin=False,
                                                             polygons=collection_polygon,
                                                             )


                fill_polygons = find_polygons_to_fill_to_measure(collection_polygon, list_points_geom_me)
                new_polygons = crop_polygons_below_surface_line(collection_polygon, list_points_geom_me)

                polygon_final = fill_polygons + new_polygons

                polygons_dict = dict(
                    initial_polygons=collection_polygon,
                    fill_polygons=fill_polygons,
                    new_polygons=new_polygons,
                )
#                plot_measure_profile(polygons_dict, list_points_geom_me, title=f'{section.name}_berm_{i}')

                modify_stix(stix_path, fill_polygons, str(i), collection_polygon, list_points_geom_me, polygons_dict)
        break  # TDOD remove this line. this is to prevent looping over all the measures within one DV

