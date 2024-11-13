from pathlib import Path

from vrtool.api import ApiRunWorkflows
from vrtool.defaults.vrtool_config import VrtoolConfig
from vrtool.orm.models import MeasureResultSection, MeasureResult, MeasurePerSection, Measure
from vrtool.orm.orm_controllers import get_all_measure_results_with_supported_investment_years, open_database
from time import time
import shutil


def rerun_database(_vr_config: VrtoolConfig, rerun_all: bool = False):
    if rerun_all:
        _vr_config.design_methods = ["Veiligheidsrendement", "Doorsnede-eisen"]

        _vr_config.T = [0, 19, 20, 25, 50, 75, 100]
        api = ApiRunWorkflows(_vr_config)
        t0 = time()
        api.run_all()
        t1 = time()
        print("IDS to import done: ", t1 - t0)
        return

    # _vr_config.design_methods = ["Specifieke doorsnede-eisen", "Veiligheidsrendement"]
    _vr_config.design_methods = ["Veiligheidsrendement", "Doorsnede-eisen"]

    # _vr_config.input_directory = path_model
    _vr_config.T = [0, 19, 20, 25, 50, 75, 100]

    t0 = time()
    api = ApiRunWorkflows(_vr_config)
    _ids_to_import = get_all_measure_results_with_supported_investment_years(_vr_config)
    t1 = time()
    print("IDS to import done: ", t1 - t0)

    # Run lenient DSN
    _vr_config.requirements_file = _vr_config.input_directory.joinpath("requirements_lenient.csv")
    api.run_optimization(optimization_name="Basisberekening_mod",
                         selected_measures_id_year=_ids_to_import)
    t2 = time()
    print("Lenient DSN done: ", t2 - t1)

def run_dsn_herverdeling_omega(_vr_config: VrtoolConfig, new_database_name: str):
    """
    Running this script requires the user to modifiy the database in place with the new omega for piping, stability and
    overflow,
    Args:
        _vr_config:
        new_database_name:

    Returns:

    """
    _vr_config.design_methods = ["Doorsnede-eisen"]
    _vr_config.input_directory = _input_model
    _vr_config.input_database_name = new_database_name
    _vr_config.T = [0, 19, 20, 25, 50, 75, 100]

    api = ApiRunWorkflows(_vr_config)
    _ids_to_import = get_all_measure_results_with_supported_investment_years(_vr_config)

    # Run lenient DSN
    api.run_optimization(optimization_name="Basisberekening_herverdeling_omega",
                         selected_measures_id_year=_ids_to_import)
    print("Herverdeling DSN done")


def run_dsn_lenient_and_stringent(_vr_config: VrtoolConfig, run_strict: bool = False):
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

    if not run_strict:
        # Run stringent DSN
        _vr_config.requirements_file = Path(_input_model).joinpath("requirements_strict.csv")
        api.run_optimization(optimization_name="Basisberekening_strict",
                             selected_measures_id_year=_ids_to_import)
        t3 = time()
        print("Stringent DSN done: ", t3 - t2)


_input_model = Path(r"C:\Users\hauth\OneDrive - Stichting Deltares\projects\VRTool\databases\41-1_test_automation\cheap")
assert _input_model.exists()
_vr_config = VrtoolConfig().from_json(Path(_input_model).joinpath("config.json"))

# copy_database(_vr_config)
# modify_cost_measure_database(multiplier=0.5, measure_type="soil reinforcement")
# rerun_database()
run_dsn_lenient_and_stringent(_vr_config)
