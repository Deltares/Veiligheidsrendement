from pathlib import Path

import numpy as np
import pandas as pd


def compare_investment_strategy(
    workbook_tc: pd.DataFrame,
    workbook_oi: pd.DataFrame,
    investment_limit: float,
    output_file_path: Path,
    measure_file_path: Path,
):
    _investment_tc = 0
    workbook_tc["dcrest"] = np.where(
        workbook_tc["dcrest"] == -999, "-", workbook_tc["dcrest"]
    )
    workbook_tc["dberm"] = np.where(
        workbook_tc["dberm"] == -999, "-", workbook_tc["dberm"]
    )
    workbook_oi["dcrest"] = np.where(
        workbook_oi["dcrest"] == -999, "-", workbook_oi["dcrest"]
    )
    workbook_oi["dberm"] = np.where(
        workbook_oi["dberm"] == -999, "-", workbook_oi["dberm"]
    )

    # Read measure data
    _read_measures = pd.read_csv(measure_file_path, delimiter=";")

    # Open a new .xlsx file in the result directory
    with pd.ExcelWriter(output_file_path) as writer:
        _df_tc = pd.DataFrame(
            data=workbook_oi["Section"].sort_values().tolist(), columns=["Dijkvak"]
        )
        _df_tc_below_limit = pd.DataFrame(
            columns=[
                "Dijkvak",
                "Prioritering",
                "TC Maatregel Tot Limiet",
                "Kosten",
                "D_Crest",
                "D_Berm",
            ]
        )
        _df_tc_above_limit = pd.DataFrame(
            columns=["Dijkvak", "TC Maatregel Na Limiet", "Kosten", "D_Crest", "D_Berm"]
        )
        _df_tc_below_limit["Dijkvak"] = _df_tc["Dijkvak"]
        _df_tc_above_limit["Dijkvak"] = _df_tc["Dijkvak"]

        for i, j in enumerate(range(len(workbook_tc["Section"]))):
            measure_id = workbook_tc["ID"][j + 1].split("+")

            # Replace the measure ID with the measure description
            if len(measure_id) == 1:
                measure_description = _read_measures[
                    _read_measures["ID"] == int(measure_id[0])
                ]["Name"].values[0]
            else:
                measure_description = (
                    _read_measures[_read_measures["ID"] == int(measure_id[0])][
                        "Name"
                    ].values[0]
                    + " + "
                    + _read_measures[_read_measures["ID"] == int(measure_id[1])][
                        "Name"
                    ].values[0]
                )

            # Count investment costs
            _investment_tc += workbook_tc["LCC"][j + 1]

            # Determine limit index
            if _investment_tc <= investment_limit:
                index = _df_tc_below_limit[
                    _df_tc_below_limit["Dijkvak"] == workbook_tc["Section"][j + 1]
                ].index
                _df_tc_below_limit.iloc[index, [2, 4, 5]] = [
                    measure_description,
                    workbook_tc["dcrest"][j + 1],
                    workbook_tc["dberm"][j + 1],
                ]
                if _df_tc_below_limit.iloc[index, 1].isnull().any().any():
                    _df_tc_below_limit.iloc[index, 1] = i + 1
                if _df_tc_below_limit.iloc[index, 3].isnull().any().any():
                    costs = 0
                else:
                    costs = _df_tc_below_limit.iloc[index, 3].values
                costs += workbook_tc["LCC"][j + 1]
                _df_tc_below_limit.iloc[index, 3] = costs
            else:
                index = _df_tc_above_limit[
                    _df_tc_above_limit["Dijkvak"] == workbook_tc["Section"][i + 1]
                ].index
                _df_tc_above_limit.iloc[index, [1, 3, 4]] = [
                    measure_description,
                    workbook_tc["dcrest"][j + 1],
                    workbook_tc["dberm"][j + 1],
                ]
                if _df_tc_above_limit.iloc[index, 2].isnull().any().any():
                    costs = 0
                else:
                    costs = _df_tc_above_limit.iloc[index, 2].values
                costs += workbook_tc["LCC"][j + 1]
                _df_tc_above_limit.iloc[index, 2] = costs

        # Merge dataframes
        _df_tc_below_limit.drop("Dijkvak", axis=1, inplace=True)
        _df_tc_above_limit.drop("Dijkvak", axis=1, inplace=True)
        _df_tc = pd.concat(
            [_df_tc, _df_tc_below_limit, _df_tc_above_limit], axis=1, ignore_index=True
        )
        _df_tc.fillna("", inplace=True)

        _df_oi = pd.DataFrame(
            {
                "Dijkvak": workbook_oi["Section"].sort_values().tolist(),
                "OI Maatregel": "",
                "Kosten": "",
                "D_Crest": "",
                "D_Berm": "",
            }
        )

        for i in range(len(workbook_oi["Section"])):
            measure_id = workbook_oi["ID"][i + 1].split("+")

            # Replace the measure ID with the measure description
            if len(measure_id) == 1:
                measure_description = _read_measures[
                    _read_measures["ID"] == int(measure_id[0])
                ]["Name"].values[0]
            else:
                measure_description = (
                    _read_measures[_read_measures["ID"] == int(measure_id[0])][
                        "Name"
                    ].values[0]
                    + " + "
                    + _read_measures[_read_measures["ID"] == int(measure_id[1])][
                        "Name"
                    ].values[0]
                )

            index = _df_oi[_df_oi["Dijkvak"] == workbook_oi["Section"][i + 1]].index
            _df_oi.iloc[index, 1:] = [
                measure_description,
                workbook_oi["LCC"][i + 1],
                workbook_tc["dcrest"][i + 1],
                workbook_tc["dberm"][i + 1],
            ]

        _df_oi.fillna("", inplace=True)

        # Write data to .xlsx file
        _sheet_name = "TC and OI Comparison"
        _df_tc.to_excel(
            writer,
            sheet_name=_sheet_name,
            header=False,
            index=False,
            startrow=1,
            engine="openpyxl",
        )
        _df_oi.to_excel(
            writer,
            sheet_name=_sheet_name,
            header=False,
            index=False,
            startrow=1,
            startcol=11,
            engine="openpyxl",
        )

        # Add Header Format
        column_header = [
            "Dijkvak",
            "Prioritering",
            "TC Maatregel Tot Limiet",
            "Kosten",
            "D_Crest",
            "D_Berm",
            "TC Maatregel Na Limiet",
            "Kosten",
            "D_Crest",
            "D_Berm",
            "",
            "Dijkvak",
            "OI Maatregel",
            "Kosten",
            "D_Crest",
            "D_Berm",
        ]
        header_format = writer.book.add_format({"bold": True})

        for col_num, value in enumerate(column_header):
            writer.sheets[_sheet_name].write(0, col_num, value, header_format)

        writer.save()
