from pathlib import Path
import pandas as pd
from vrtool.decision_making.measures.measure_base import MeasureBase
from vrtool.decision_making.measures import (
    CustomMeasure,
    DiaphragmWallMeasure,
    SoilReinforcementMeasure,
    StabilityScreenMeasure,
    VerticalGeotextileMeasure,
)
from vrtool.flood_defence_system.dike_section import DikeSection

from vrtool.orm.io.importers.orm_importer_protocol import OrmImporterProtocol
from vrtool.defaults.vrtool_config import VrtoolConfig
from vrtool.orm.models.measure import Measure
from vrtool.orm.models.orm_base_model import OrmBaseModel
from vrtool.orm.models.standard_measure import StandardMeasure


class MeasureImporter(OrmImporterProtocol):
    
    _vrtool_config: VrtoolConfig
    dike_section: DikeSection
    berm_step: list[int]
    t_0: int
    geometry_plot: bool
    unit_costs: dict


    def __init__(self, vrtool_config: VrtoolConfig, dike_section: DikeSection) -> None:
        self._vrtool_config = vrtool_config
        self.dike_section = dike_section
        self.berm_step = vrtool_config.berm_step
        self.t_0 = vrtool_config.t_0
        self.geometry_plot = vrtool_config.geometry_plot
        self.unit_costs = vrtool_config.unit_costs


    def _set_standard_measure_values(self, measure: MeasureBase, orm_measure: StandardMeasure) -> None:
        measure.config = self._vrtool_config
        measure.crest_step = orm_measure.crest_step
        measure.berm_step = self.berm_step
        measure.t_0 = self.t_0
        measure.geometry_plot = self.geometry_plot
        measure.unit_costs = self.unit_costs
        measure.parameters = {}
        measure.parameters["Direction"] = orm_measure.direction
        measure.parameters["StabilityScreen"] = "yes" if orm_measure.stability_screen else "no"
        measure.parameters["dcrest_min"] = None
        measure.parameters["dcrest_max"] = orm_measure.max_crest_increase
        measure.parameters["max_outward"] = orm_measure.max_outward_reinforcement
        measure.parameters["max_inward"] = orm_measure.max_inward_reinforcement
        measure.parameters["ID"] = orm_measure.get_id()

    def _import_soil_reinforcement_measure(self, orm_measure: Measure) -> SoilReinforcementMeasure:
        _measure = SoilReinforcementMeasure()
        self._set_standard_measure_values(_measure, orm_measure.standard_measure)
        _measure.parameters["Type"] = "Soil Reinforcement"

        return _measure

    def _import_diaphragm_wall_measure(self, orm_measure: Measure) -> DiaphragmWallMeasure:
        _measure = DiaphragmWallMeasure()
        self._set_standard_measure_values(_measure, orm_measure.standard_measure)
        _measure.parameters["Type"] = "Diaphragm Wall"

    def _import_stability_screen_measure(self, orm_measure: Measure) -> StabilityScreenMeasure:
        _measure = StabilityScreenMeasure()
        self._set_standard_measure_values(_measure, orm_measure.standard_measure)
        _measure.parameters["Type"] = "Stability Screen"

    def _import_vertical_geotextile_measure(self, orm_measure: Measure) -> VerticalGeotextileMeasure:
        _measure = VerticalGeotextileMeasure()
        self._set_standard_measure_values(_measure, orm_measure.standard_measure)
        _measure.parameters["Type"] = "Vertical Geotextile"

    def _import_custom_measure(self) -> CustomMeasure:
        raise NotImplementedError()

    def _import_standard_measure(self, orm_measure: StandardMeasure) -> MeasureBase:
        _measure_type = orm_measure.measure.measure_type.name.lower()
        if _measure_type == "Soil Reinforcement":
            return self._import_soil_reinforcement_measure(orm_measure)
        elif _measure_type == "Diaphragm Wall":
            return self._import_diaphragm_wall_measure(orm_measure)
        elif _measure_type == "Stability Screen":
            return self._import_stability_screen_measure(orm_measure)
        elif _measure_type == "Vertical Geotextile":
            return self._import_vertical_geotextile_measure(orm_measure)
        else:
            raise NotImplementedError("No import available for {}.".format(_measure_type))

    def import_orm(self, orm_model: Measure) -> MeasureBase:
        _measure_type = orm_model.measure_type.name.lower()
        if _measure_type == "custom":
            return self._import_custom_measure()
        return self._import_standard_measure(orm_model.standard_measure)
