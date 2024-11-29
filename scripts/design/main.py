from pathlib import Path
import pandas as pd

from scripts.design.combine_omega_length_effect import run_combination_single_database, create_summary_results_csv
from scripts.design.modify_inputs import copy_database, modify_beta_measure_database, modify_cost_measure_database
from scripts.design.plot_design import plot_histogram_metrics, plot_histogram_metrics_respective_to_base
from scripts.design.run_vrtool_specific import rerun_database
from scripts.design.sensitivity import read_sensitivity_results_and_plot, \
    compare_lenient_requirement_tables, plot_sensitivity_box_plot_beta_all_runs, make_boxplot_sensitivity_per_sections, \
    create_comparison_section_table
from vrtool.defaults.vrtool_config import VrtoolConfig

_input_model = Path(
    r"C:\Users\hauth\OneDrive - Stichting Deltares\projects\VRTool\databases\38-1\automation\test")
assert _input_model.exists()


# res_list = []
# N = 6
# for i, multiplier in enumerate([0.5, 0.75, 1.25, 1.5, 2.0, 2.5, 3]):
#     _vr_config = VrtoolConfig().from_json(Path(_input_model).joinpath("config.json"))
#     _vr_config.input_directory = _input_model
#     copy_database(_vr_config, f"modified_beta_{i}")
#     # modify_beta_measure_database(_vr_config, measure_type="soil reinforcement", mechanism="StabilityInner", std_beta=1.0)
#     modify_cost_measure_database(_vr_config, measure_type="soil reinforcement + screen", multiplier=multiplier)
#     # rerun_database(_vr_config, rerun_all=False, run_name=f"modified_beta_{i}")
#     try:
#         res, vrm_optimization_steps, df_combinations_results, p_max = run_combination_single_database(_vr_config.input_directory.joinpath(_vr_config.input_database_name), run_id=3)
#         res_list.append(res)
#     except:
#         continue
#
#     print(f"Finished run {i} of {N}")
#
# # convert to csv
# df = pd.DataFrame(res_list)
# df.to_csv(_input_model.joinpath("results_sensitivity_analysis_modified_piping_beta_VZG.csv"))
#
# create_summary_results_csv(dir_path=Path(r"N:\Projects\11209000\11209353\B. Measurements and calculations\Handleiding & handreiking\sensitivity_analysis\10-3\modified_beta_stability_beoordeling"),
#                            filename="results_sensitivity_beta_stab_beoordeling",
#                            run_id=3)

run_combination_single_database(Path(r"N:\Projects\11209000\11209353\B. Measurements and calculations\Handleiding & handreiking\sensitivity_analysis\modified_beta_stability_beoordeling\38-1_basis_0.db"
                                     ), plot=True, run_id=1)



# create_csv_runs_summary()
# df = read_sensitivity_results_and_plot(Path(
#     # r"C:\Users\hauth\OneDrive - Stichting Deltares\projects\VRTool\databases\38-1\automation\results_sensitivity_analysis_38-1.csv"),
#     r"N:\Projects\11209000\11209353\B. Measurements and calculations\Handleiding & handreiking\sensitivity_analysis\modified_cost_VZG/results_sensitivity_cost_VZG.csv"),
#     Path(r"C:\Users\hauth\OneDrive - Stichting Deltares\projects\VRTool\databases\38-1\automation\38-1_basis_0.db")
# )
# df = pd.read_csv(r"N:\Projects\11209000\11209353\B. Measurements and calculations\Handleiding & handreiking\sensitivity_analysis\modified_cost_VZG\results_sensitivity_cost_VZG.csv")
# plot_histogram_metrics(df, True, save=True)
# plot_histogram_metrics_respective_to_base(df, False, save=True)
# compare_lenient_requirement_tables()

# compare_lenient_requirement_tables()
path_dir = Path(r"N:\Projects\11209000\11209353\B. Measurements and calculations\Handleiding & handreiking\sensitivity_analysis\modified_cost_VZG")
# df = plot_sensitivity_box_plot_beta_all_runs(path_dir, mechanism="piping")
# df = pd.read_csv(path_dir / "beta_piping_per_section.csv")
# make_boxplot_sensitivity_per_sections(df)

create_comparison_section_table(path_dir, run_id=3)
# Workflow:
# 1. run create_summary_results_csv to make the csv summary file of the case
# 2. Manually add the base case to the csv file at the last row!
# 3. Run read_sensitivity_results_and_plot to plot the sensitivity analysis
# 4. Run histogram metrics to plot the histograms