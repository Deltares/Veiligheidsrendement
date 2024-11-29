from pathlib import Path

from vrtool.defaults.vrtool_config import VrtoolConfig
from vrtool.orm.models import ComputationScenarioParameter
from vrtool.orm.orm_controllers import open_database

path_base_dir = Path(r"N:\Projects\11209000\11209353\B. Measurements and calculations\Handleiding & handreiking\sensitivity_analysis")
path_true_dir = Path(r"N:\Projects\11209000\11209353\B. Measurements and calculations\008 - Resultaten Proefvlucht\Alle_Databases\38-1\oude_teenlijn")

_vr_config = VrtoolConfig().from_json(Path(path_base_dir).joinpath("config.json"))

_connected_db = open_database(_vr_config.input_database_path)

query = ComputationScenarioParameter.select().where(ComputationScenarioParameter.parameter == "beta")

