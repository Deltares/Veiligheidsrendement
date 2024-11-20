import itertools
from pathlib import Path
import copy

import numpy as np
import pandas as pd

from scripts.design.combine_omega_length_effect import run_single_database
from scripts.design.create_requirement_files import get_traject_requirements_per_sections, \
    create_requirement_files_lenient_and_strict
from scripts.design.plot_design import plot_sensitivity
from openpyxl import Workbook
from openpyxl.styles import PatternFill
from openpyxl.utils.dataframe import dataframe_to_rows



def create_csv_runs_summary():
    path_dir = Path(r"C:\Users\hauth\OneDrive - Stichting Deltares\projects\VRTool\databases\38-1\automation")
    # loop over all  .db
    res_list = []
    for db_path in path_dir.glob("*.db"):
        print(db_path, "processing ...")

        res, vrm_optimization_steps, df_combinations_results, p_max = run_single_database(db_path)
        res_list.append(res)

    # convert to csv
    df = pd.DataFrame(res_list)
    df.to_csv(path_dir.joinpath("results_sensitivity_analysis_38-1.csv"))

    pass


def read_sensitivity_results_and_plot(csv_result_path: Path, db_path):
    df_sensitivity = pd.read_csv(csv_result_path)
    _, vrm_optimization_steps, _, p_max = run_single_database(db_path)
    plot_sensitivity(df_sensitivity, vrm_optimization_steps, p_max)


from openpyxl.styles import PatternFill
from openpyxl.utils.dataframe import dataframe_to_rows
from pathlib import Path
import pandas as pd
from openpyxl import Workbook

def compare_lenient_requirement_tables():
    path_dir = Path(r"C:\Users\hauth\OneDrive - Stichting Deltares\projects\VRTool\databases\38-1\automation")
    wb = Workbook()
    ws_base = wb.active
    ws_base.title = "Base Case"

    base_lenient_df = None
    highlight_fill = PatternFill(start_color="FFEE08", end_color="FFEE08", fill_type="solid")  # Yellow fill

    summary_data = []  # To collect counts of differences for the summary sheet

    for idx, db_path in enumerate(path_dir.glob("*.db")):
        print(db_path, "processing ...")
        params = {
            "name": "38-1",
            "path": db_path,
            "eis": 1 / 10000,
            "l_traject": 28900,
            "revetment": False,
            "meet_eis": True,
        }
        requirements_per_section = get_traject_requirements_per_sections(params)
        lenient_df, _ = create_requirement_files_lenient_and_strict(requirements_per_section, params)

        if idx == 0:
            # Assume first file is the base case
            base_lenient_df = lenient_df
            for r in dataframe_to_rows(base_lenient_df, index=False, header=True):
                ws_base.append(r)
            # Initialize summary data structure with 0 differences
            summary_data = [{"Row": i + 1, "OVERFLOW": 0, "PIPING": 0, "STABILITY_INNER": 0} for i in range(len(base_lenient_df))]
        else:
            # Compare with the base case
            ws_sim = wb.create_sheet(title=f"Simulation {idx}")
            for r_idx, row in enumerate(dataframe_to_rows(lenient_df, index=False, header=True), start=1):
                ws_sim.append(row)
                if r_idx == 1:  # Skip header row for highlighting
                    continue
                for c_idx, col_name in enumerate(lenient_df.columns, start=1):
                    cell = ws_sim.cell(row=r_idx, column=c_idx)
                    base_value = base_lenient_df.iloc[r_idx - 2, c_idx - 1]  # Adjust for header row
                    current_value = lenient_df.iloc[r_idx - 2, c_idx - 1]
                    if col_name in ["OVERFLOW", "PIPING", "STABILITY_INNER"] and base_value != current_value:
                        cell.fill = highlight_fill  # Highlight cell with a different value
                        # Increment difference count for this row and column
                        summary_data[r_idx - 2][col_name] += 1

    # Add a summary sheet
    ws_summary = wb.create_sheet(title="Summary")
    summary_df = pd.DataFrame(summary_data)
    for r in dataframe_to_rows(summary_df, index=False, header=True):
        ws_summary.append(r)

    # Save workbook
    output_file = path_dir / "comparison_results_with_summary.xlsx"
    wb.save(output_file)
    print(f"Comparison saved to {output_file}")


