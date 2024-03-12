from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from vrtool.common.enums import MechanismEnum
from vrtool.optimization.measures.section_as_input import SectionAsInput
from vrtool.probabilistic_tools.probabilistic_functions import beta_to_pf, pf_to_beta

def modify_kunstwerk_initial_safety(results_measures: list[SectionAsInput], structure_name: str, file_path: Path) -> list[SectionAsInput]:
    #load from file path
    kw_safety = pd.read_excel(file_path, sheet_name='HuidigeVeiligheid',index_col=0)

    #modify the section_reliability dataframe
    for section_measures in results_measures:
        if section_measures.section_name == structure_name:
            print('found')
            for _prob_entry in section_measures.initial_assessment.probabilities:
                if _prob_entry.mechanism == MechanismEnum.OVERFLOW:
                    _prob_entry.probability = beta_to_pf(kw_safety.loc['HTKW','beta'])
                elif _prob_entry.mechanism == MechanismEnum.PIPING:
                    _prob_entry.probability = beta_to_pf(kw_safety.loc['BSKW','beta'])
                elif _prob_entry.mechanism == MechanismEnum.STABILITY_INNER:
                    pf_pkw = beta_to_pf(kw_safety.loc['PKW','beta'])
                    pf_stkwp = beta_to_pf(kw_safety.loc['STKWp','beta'])
                    pf_combined = 1-(1-pf_pkw)*(1-pf_stkwp)
                    _prob_entry.probability = pf_combined

    return results_measures
def modify_kunstwerkmaatregel(results_measures: list[SectionAsInput], structure_name: str, file_path: Path) -> list[SectionAsInput]:
    #load from file path
    kw_measures = pd.read_excel(file_path, sheet_name='Maatregelen')
    _costs_list = list(kw_measures['kosten'])
    for _section_measures in results_measures:
        if _section_measures.section_name == structure_name:
            #modify cost of sh measure to 0. And add the right htkw pf
            for _measure in _section_measures.sh_measures:
                    _measure.cost = 0
                    for _prob_entry in _measure.mechanism_year_collection.probabilities:
                        if _prob_entry.mechanism == MechanismEnum.OVERFLOW:
                            _prob_entry.probability = beta_to_pf(kw_measures.loc[0,'HTKW'])
            #add the new measures as sg measures
            for count, _measure in enumerate(_section_measures.sg_measures):
                _measure.cost = _costs_list.pop(0)
                _measure.start_cost = 0.
                for _prob_entry in _measure.mechanism_year_collection.probabilities:
                    if _prob_entry.mechanism == MechanismEnum.PIPING:
                        _prob_entry.probability = beta_to_pf(kw_measures.loc[count,'BSKW'])
                    elif _prob_entry.mechanism == MechanismEnum.STABILITY_INNER:
                        pf_pkw = beta_to_pf(kw_measures.loc[count,'PKW'])
                        pf_stkwp = beta_to_pf(kw_measures.loc[count,'STKWp'])
                        pf_combined = 1-(1-pf_pkw)*(1-pf_stkwp)
                        _prob_entry.probability = pf_combined
    return results_measures