from typing import Type

from vrtool.decision_making.measures import (
    CustomMeasure,
    DiaphragmWallMeasure,
    SoilReinforcementMeasure,
    StabilityScreenMeasure,
    VerticalGeotextileMeasure,
)
from vrtool.decision_making.measures.custom_measure import CustomMeasure
from vrtool.decision_making.measures.measure_protocol import MeasureProtocol
from vrtool.defaults.vrtool_config import VrtoolConfig
from vrtool.flood_defence_system.dike_section import DikeSection
from vrtool.orm.io.importers.orm_importer_protocol import OrmImporterProtocol
from vrtool.orm.models.custom_measure import CustomMeasure as OrmCustomMeasure
from vrtool.orm.models.measure import Measure as OrmMeasure
from vrtool.orm.models.standard_measure import StandardMeasure


class MeasureImporter(OrmImporterProtocol):

    _config: VrtoolConfig
    berm_step: list[int]
    t_0: int
    geometry_plot: bool
    unit_costs: dict

    def __init__(self, vrtool_config: VrtoolConfig) -> None:
        if not vrtool_config:
            raise ValueError("VrtoolConfig not provided.")

        self._config = vrtool_config
        self.berm_step = vrtool_config.berm_step
        self.t_0 = vrtool_config.t_0
        self.geometry_plot = vrtool_config.geometry_plot
        self.unit_costs = vrtool_config.unit_costs

    def _set_base_values(self, measure: MeasureProtocol):
        measure.config = self._config
        measure.berm_step = self.berm_step
        measure.t_0 = self.t_0
        measure.geometry_plot = self.geometry_plot
        measure.unit_costs = self.unit_costs
        measure.parameters = {}

    def _get_standard_measure(
        self, measure_type: Type[MeasureProtocol], orm_measure: StandardMeasure
    ) -> MeasureProtocol:
        _measure = measure_type()
        self._set_base_values(_measure)
        _measure.crest_step = orm_measure.crest_step
        _measure.parameters["Name"] = orm_measure.measure.name
        _measure.parameters["Type"] = orm_measure.measure.measure_type.name
        _measure.parameters["Class"] = orm_measure.measure.combinable_type.name
        _measure.parameters["Direction"] = orm_measure.direction
        _measure.parameters["StabilityScreen"] = (
            "yes" if orm_measure.stability_screen else "no"
        )

        # dcrest_min is a fix value to 0. For now we keep the property in the params dictionary. Eventually can be removed.
        _measure.parameters["dcrest_min"] = 0
        _measure.parameters["dcrest_max"] = orm_measure.max_crest_increase
        _measure.parameters["max_outward"] = orm_measure.max_outward_reinforcement
        _measure.parameters["max_inward"] = orm_measure.max_inward_reinforcement
        _measure.parameters["year"] = orm_measure.measure.year
        _measure.parameters["P_solution"] = orm_measure.prob_of_solution_failure
        _measure.parameters[
            "Pf_solution"
        ] = orm_measure.failure_probability_with_solution
        _measure.parameters["ID"] = orm_measure.get_id()

        return _measure

    def _import_custom_measure(self, orm_measure: OrmCustomMeasure) -> CustomMeasure:
        _measure = CustomMeasure()
        self._set_base_values(_measure)
        _measure.measures = {}
        _measure.measures["Cost"] = orm_measure.cost
        _measure.measures["Reliability"] = orm_measure.beta
        _measure.parameters["year"] = orm_measure.year

        for _measure_parameter in orm_measure.custom_parameters:
            _measure.parameters[_measure_parameter.parameter] = _measure_parameter.value

        _measure.parameters["Class"] = orm_measure.measure.combinable_type.name
        _measure.parameters["Name"] = orm_measure.measure.name
        _measure.parameters["ID"] = orm_measure.get_id()

        return _measure

    def _import_standard_measure(self, orm_measure: StandardMeasure) -> MeasureProtocol:
        _mapping_types = {
            "soil reinforcement": SoilReinforcementMeasure,
            "soil reinforcement with stability screen": SoilReinforcementMeasure,
            "diaphragm wall": DiaphragmWallMeasure,
            "stability screen": StabilityScreenMeasure,
            "vertical geotextile": VerticalGeotextileMeasure,
        }

        _found_type = _mapping_types.get(
            orm_measure.measure.measure_type.name.lower().strip(), None
        )
        if not _found_type:
            raise NotImplementedError(
                "No import available for {}.".format(
                    orm_measure.measure.measure_type.name
                )
            )

        return self._get_standard_measure(_found_type, orm_measure)

    def import_orm(self, orm_model: OrmMeasure) -> MeasureProtocol:

        if not orm_model:
            raise ValueError(f"No valid value given for {OrmMeasure.__name__}.")

        _measure_type = orm_model.measure_type.name.lower()
        if _measure_type == "custom":
            return self._import_custom_measure(orm_model.custom_measures.select().get())
        return self._import_standard_measure(orm_model.standard_measure.select().get())
