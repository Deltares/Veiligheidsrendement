from pathlib import Path
import pandas as pd

from scripts.design.combine_omega_length_effect import run_single_database
from scripts.design.modify_inputs import copy_database, modify_beta_measure_database
from scripts.design.run_vrtool_specific import rerun_database
from vrtool.defaults.vrtool_config import VrtoolConfig


_input_model = Path(
    r"C:\Users\hauth\OneDrive - Stichting Deltares\projects\VRTool\databases\38-1\test")
assert _input_model.exists()
_vr_config = VrtoolConfig().from_json(Path(_input_model).joinpath("config.json"))
_vr_config.input_directory = _input_model


res_list = []
N = 1
for i in range(1, N+1):
    copy_database(_vr_config, f"modified_beta_{i}")
    modify_beta_measure_database(_vr_config, measure_type="soil reinforcement", mechanism="StabilityInner", std_beta=1.0)
    rerun_database(_vr_config, rerun_all=False)
    res, vrm_optimization_steps, df_combinations_results, p_max = run_single_database(_vr_config.input_directory.joinpath(_vr_config.input_database_name))
    res_list.append(res)

    print(f"Finished run {i} of {N}")

# convert to csv
df = pd.DataFrame(res_list)
df.to_csv(_input_model.joinpath("results_sensitivity_analysis_38-1.csv"))


# modify_cost_measure_database(_vr_config, multiplier=2, measure_type="soil reinforcement")
# modify_initial_beta_stability(_vr_config, std_beta=1.0)
# rerun_database(_vr_config, rerun_all=True)
# run_dsn_lenient_and_stringent(_vr_config, run_strict=False)


# plot_combined_measure_figure(df_combinations=df_combinations_results,
#                              vrm_optimization_steps=vrm_optimization_steps,
#                              vrm_optimum_point=(res["vrm_optimal_point_cost"], res["vrm_optimal_point_pf"]),
#                              least_expensive_combination=(res["cheapest_combination_cost"], res["cheapest_combination_pf"]),
#                              dsn_point=(res["dsn_point_cost"], res["dsn_point_pf"]),
#                              p_max=p_max)


