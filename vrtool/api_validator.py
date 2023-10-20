from vrtool.defaults.vrtool_config import VrtoolConfig
from vrtool.orm.orm_controllers import open_database
import vrtool.orm.models as orm_models

class apiValidator:
    def get_measure_result_ids(
        self, valid_vrtool_config: VrtoolConfig
    ) -> list[int]:
        _connected_db = open_database(valid_vrtool_config.input_database_path)
        _id_list = [mr for mr in orm_models.MeasureResult.select()]
        _connected_db.close()
        return _id_list

    def get_measure_result_with_investment_year(
        self,
        measures_results: list[orm_models.MeasureResult],
    ) -> list[tuple[int, int]]:
        _measure_result_with_year_list = []
        for _measure_result in measures_results:
            if _measure_result.measure_per_section.measure.year == 20:
                # We do not want measures that have a year variable >0 initially, as then the interpolation is messed up.
                continue
            
            # All will get at least year 0.
            _measure_result_with_year_list.append((_measure_result.get_id(), 0))
            if "Soil reinforcement" in _measure_result.measure_per_section.measure.measure_type.name:
                # For those of type "Soil reinforcement" we also add year 20.
                _measure_result_with_year_list.append((_measure_result.get_id(), 20))
        return _measure_result_with_year_list
