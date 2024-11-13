import itertools
from pathlib import Path
import copy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scripts.design.deltares_colors import colors
from scripts.design.plot_design import plot_combined_measure_figure
from scripts.design.utils import get_measures_for_all_sections, add_combined_vzg_soil, get_traject_probs, \
    calculate_cost, get_target_beta_grid, get_cost_traject_pf_combinations
from scripts.postprocessing.database_access_functions import get_overview_of_runs, get_optimization_steps_for_run_id
from scripts.postprocessing.database_analytics import get_minimal_tc_step

from vrtool.common.enums import MechanismEnum
from vrtool.orm.models import SectionData, MechanismPerSection, Mechanism, MeasureResultMechanism, MeasureResultSection, \
    MeasureResult, MeasurePerSection, Measure, MeasureType, DikeTrajectInfo
from vrtool.probabilistic_tools.probabilistic_functions import pf_to_beta, beta_to_pf

db_path = Path(
    # r"C:\Users\hauth\OneDrive - Stichting Deltares\projects\VRTool\databases\41-1_test_automation\41-1_database_origineel.db")
    r"C:\Users\hauth\OneDrive - Stichting Deltares\projects\VRTool\databases\41-1_test_automation\automation\41-1_database_origineel_modified_stability.db")
    # r"C:\Users\hauth\OneDrive - Stichting Deltares\projects\VRTool\databases\38-1\38-1_basis.db")
# db_path = Path(r"c:\Users\klerk_wj\OneDrive - Stichting Deltares\00_Projecten\11_VR_HWBP\05_Handreiking\casus_ontwerpen\53-1\53-1_vakindeling1.db")
has_revetment = False
LE = False

t_design = 50

_runs_overview = get_overview_of_runs(db_path)
# Get the traject probabilities over time for every step in the optimization process
traject_probs = get_traject_probs(db_path)
p_max = DikeTrajectInfo.get(DikeTrajectInfo.id == 1).p_max * 0.52

# Put all the measures in a DataFrame for DSN
measures_df_db = get_measures_for_all_sections(db_path, t_design)
measures_df = add_combined_vzg_soil(measures_df_db)
# sort measures_df by section_id and measure_result
measures_df.sort_values(by=['section_id', 'measure_result'], inplace=True)

# as probabilities in the df are to be interpreted as probabilities per section, we need to also compute the corresponding cross-section probabilities

# get the section lengths from the database
section_lengths = pd.DataFrame([(section.id, section.section_length) for section in SectionData.select()],
                               columns=['section_id', 'section_length'])
section_lengths.set_index('section_id', inplace=True)
# merge lengths to measures_df
measures_df_with_dsn = measures_df.merge(section_lengths, left_on='section_id', right_index=True)
measures_df_with_dsn['Overflow_dsn'] = measures_df_with_dsn['Overflow']
if LE:
    N_piping = measures_df_with_dsn['section_length'] / 300
    N_piping = N_piping.apply(lambda x: max(x, 1))
    measures_df_with_dsn['Piping_dsn'] = pf_to_beta(beta_to_pf(measures_df_with_dsn['Piping']) / N_piping)
    N_stability = measures_df_with_dsn['section_length'] / 50
    N_stability = N_stability.apply(lambda x: max(x, 1))
    measures_df_with_dsn['StabilityInner_dsn'] = pf_to_beta(
        beta_to_pf(measures_df_with_dsn['StabilityInner']) / N_stability)
else:
    measures_df_with_dsn['Piping_dsn'] = measures_df_with_dsn['Piping']
    measures_df_with_dsn['StabilityInner_dsn'] = measures_df_with_dsn['StabilityInner']
measures_df_with_dsn.head()

##############

N_omega = [2., 4., 8., 16., 32.]
N_LE = [5., 10., 20., 40., 50.]
target_beta_grid = get_target_beta_grid(N_omega, N_LE)

cost, pf_traject = get_cost_traject_pf_combinations(target_beta_grid, measures_df_with_dsn)
# find the indices in the optimization steps where the total_risk is the same as the step before


# VRM

optimization_steps = get_optimization_steps_for_run_id(db_path, 1)
considered_tc_step = get_minimal_tc_step(optimization_steps) - 1

# find index where traject_probs[0][0] == 50
ind_2075 = np.where(np.array(traject_probs[0][0]) == 50)[0][0]
pf_2075 = [traject_probs[i][1][ind_2075] for i in range(len(traject_probs))]
cost_vrm = [optimization_steps[i]['total_lcc'] for i in range(len(traject_probs))]

print(considered_tc_step)

vrm_optimum_cost = optimization_steps[considered_tc_step - 1]['total_lcc']
vrm_optimum_pf = traject_probs[considered_tc_step - 1][1][ind_2075]
print(vrm_optimum_cost, vrm_optimum_pf)

plot_combined_measure_figure(cost=cost,
                             pf_traject=pf_traject,
                             cost_vrm=cost_vrm,
                             pf_2075_vrm=pf_2075,
                             cost_vrm_filtered=[],
                             pf_2075_vrm_filtered=[],
                             vrm_optimum_point=(vrm_optimum_cost, vrm_optimum_pf),
                             p_max=p_max)
