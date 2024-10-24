from pathlib import Path

from vrtool.api import ApiRunWorkflows
from vrtool.defaults.vrtool_config import VrtoolConfig
from vrtool.orm.orm_controllers import get_all_measure_results_with_supported_investment_years
from time import time



assert _input_model.exists()
vr_config = VrtoolConfig().from_json(Path(_input_model).joinpath("config.json"))

def run_dsn_lenient_and_stringent(_vr_config: VrtoolConfig):
    _vr_config.design_methods = ["Specifieke doorsnede-eisen"]
    _vr_config.input_directory = _input_model
    _vr_config.T = [0, 19, 20, 25, 50, 75, 100]

    t0 = time()
    api = ApiRunWorkflows(_vr_config)
    _ids_to_import = get_all_measure_results_with_supported_investment_years(_vr_config)
    t1 = time()
    print("IDS to import done: ", t1 - t0)


    # Run lenient DSN
    _vr_config.requirements_file = Path(_input_model).joinpath("requirements_lenient.csv")
    api.run_optimization(optimization_name="Basisberekening_lenient",
                         selected_measures_id_year=_ids_to_import)
    t2 = time()
    print("Lenient DSN done: ", t2 - t1)

    # Run stringent DSN
    _vr_config.requirements_file = Path(_input_model).joinpath("requirements_strenger.csv")
    api.run_optimization(optimization_name="Basisberekening_stringent",
                         selected_measures_id_year=_ids_to_import)
    t3 = time()
    print("Stringent DSN done: ", t3 - t2)

def run_dsn_herverdeling_omega(_vr_config: VrtoolConfig, database_name: str):
    _vr_config.design_methods = ["Doorsnede-eisen"]
    _vr_config.input_directory = _input_model
    _vr_config.input_database_name = database_name
    _vr_config.T = [0, 19, 20, 25, 50, 75, 100]

    api = ApiRunWorkflows(_vr_config)
    _ids_to_import = get_all_measure_results_with_supported_investment_years(_vr_config)

    # Run lenient DSN
    api.run_optimization(optimization_name="Basisberekening_herverdeling_omega",
                         selected_measures_id_year=_ids_to_import)
    print("Herverdeling DSN done")




### /!\ Make sure you made a backup database before running this script /!\ ###

# _input_model = Path(r"C:\Users\hauth\OneDrive - Stichting Deltares\projects\VRTool\databases\41-1")
_input_model = Path(r"C:\Users\hauth\OneDrive - Stichting Deltares\projects\VRTool\databases\10-2")

run_dsn_lenient_and_stringent(vr_config)
# run_dsn_herverdeling_omega(vr_config, "database_10-3_herverdeling.sqlite")
