from typing import Type

from vrtool.common.enums.measure_type_enum import MeasureTypeEnum
from vrtool.common.measure_unit_costs import MeasureUnitCosts
from vrtool.decision_making.measures import (
    CustomMeasure,
    DiaphragmWallMeasure,
    RevetmentMeasure,
    SoilReinforcementMeasure,
    StabilityScreenMeasure,
    VerticalPipingSolutionMeasure,
)
from vrtool.decision_making.measures.measure_protocol import MeasureProtocol
from vrtool.decision_making.measures.standard_measures.wall_measures.anchored_sheetpile_measure import (
    AnchoredSheetpileMeasure,
)
from vrtool.defaults.vrtool_config import VrtoolConfig
from vrtool.orm.io.importers.orm_importer_protocol import OrmImporterProtocol
from vrtool.orm.models.custom_measure import CustomMeasure as OrmCustomMeasure
from vrtool.orm.models.measure import Measure as OrmMeasure
from vrtool.orm.models.standard_measure import StandardMeasure


class MeasureImporter(OrmImporterProtocol):

    _config: VrtoolConfig
    berm_step: list[int]
    t_0: int
    unit_costs: MeasureUnitCosts

    def __init__(self, vrtool_config: VrtoolConfig) -> None:
        if not vrtool_config:
            raise ValueError("VrtoolConfig not provided.")

        self._config = vrtool_config
        self.berm_step = vrtool_config.berm_step
        self.t_0 = vrtool_config.t_0
        self.unit_costs = vrtool_config.unit_costs

    def _set_base_values(self, measure: MeasureProtocol):
        measure.config = self._config
        measure.berm_step = self.berm_step
        measure.t_0 = self.t_0
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
        _measure.parameters[
            "transition_level_increase_step"
        ] = orm_measure.transition_level_increase_step
        _measure.parameters["max_pf_factor_block"] = orm_measure.max_pf_factor_block
        _measure.parameters["n_steps_block"] = orm_measure.n_steps_block
        _measure.parameters["ID"] = orm_measure.get_id()

        return _measure

    def _import_standard_measure(self, orm_measure: StandardMeasure) -> MeasureProtocol:
        _mapping_types = {
            MeasureTypeEnum.SOIL_REINFORCEMENT.get_old_name(): SoilReinforcementMeasure,
            MeasureTypeEnum.SOIL_REINFORCEMENT_WITH_STABILITY_SCREEN.get_old_name(): SoilReinforcementMeasure,
            MeasureTypeEnum.DIAPHRAGM_WALL.get_old_name(): DiaphragmWallMeasure,
            MeasureTypeEnum.ANCHORED_SHEETPILE.get_old_name(): AnchoredSheetpileMeasure,
            MeasureTypeEnum.STABILITY_SCREEN.get_old_name(): StabilityScreenMeasure,
            MeasureTypeEnum.VERTICAL_PIPING_SOLUTION.get_old_name(): VerticalPipingSolutionMeasure,
            MeasureTypeEnum.REVETMENT.get_old_name(): RevetmentMeasure,
        }

        _found_type = _mapping_types.get(orm_measure.measure.measure_type.name, None)
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

        _measure_type = orm_model.measure_type.name
        if _measure_type == MeasureTypeEnum.CUSTOM.get_old_name():
            raise NotImplementedError(
                "Custom measures are not supported by this importer."
            )
        return self._import_standard_measure(orm_model.standard_measure.select().get())
