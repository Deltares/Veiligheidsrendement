from pathlib import Path
import pandas as pd

from scripts.design.combine_omega_length_effect import run_single_database
from scripts.design.modify_inputs import copy_database, modify_beta_measure_database
from scripts.design.run_vrtool_specific import rerun_database
from scripts.design.sensitivity import create_csv_runs_summary, read_sensitivity_results_and_plot, \
    compare_lenient_requirement_tables
from vrtool.defaults.vrtool_config import VrtoolConfig

_input_model = Path(
    r"C:\Users\hauth\OneDrive - Stichting Deltares\projects\VRTool\databases\38-1\automation\test")
assert _input_model.exists()


res_list = []
N = 20
for i in range(0, N):
    _vr_config = VrtoolConfig().from_json(Path(_input_model).joinpath("config.json"))
    _vr_config.input_directory = _input_model
    copy_database(_vr_config, f"modified_beta_{i}")
    modify_beta_measure_database(_vr_config, measure_type="soil reinforcement", mechanism="StabilityInner", std_beta=1.0)
    rerun_database(_vr_config, rerun_all=False, run_name=f"modified_beta_{i}")
    try:
        res, vrm_optimization_steps, df_combinations_results, p_max = run_single_database(_vr_config.input_directory.joinpath(_vr_config.input_database_name), run_id=3)
        res_list.append(res)
    except:
        continue

    print(f"Finished run {i} of {N}")
#
# # convert to csv
# df = pd.DataFrame(res_list)
# df.to_csv(_input_model.joinpath("results_sensitivity_analysis_modified_piping_beta_VZG.csv"))


# run_single_database(_vr_config.input_directory.joinpath(_vr_config.input_database_name), plot=True)


# create_csv_runs_summary()
# read_sensitivity_results_and_plot(Path(
#     # r"C:\Users\hauth\OneDrive - Stichting Deltares\projects\VRTool\databases\38-1\automation\results_sensitivity_analysis_38-1.csv"),
#     r"C:\Users\hauth\OneDrive - Stichting Deltares\projects\VRTool\databases\38-1\automation\modified_beta_stability_soil_reinforcement\results_sensitivity_analysis_modified_stab_beta_soilReinf.csv"),
#     Path(r"C:\Users\hauth\OneDrive - Stichting Deltares\projects\VRTool\databases\38-1\automation\38-1_basis_0.db")
# )
# compare_lenient_requirement_tables()
