from vrtool.failure_mechanisms.mechanism_input import MechanismInput
from vrtool.failure_mechanisms.revetment.relation_grass_revetment import (
    RelationGrassRevetment,
)
from vrtool.failure_mechanisms.revetment.revetment_data_class import RevetmentDataClass
from vrtool.failure_mechanisms.revetment.slope_part import SlopePartProtocol
from vrtool.orm.io.importers.orm_importer_protocol import OrmImporterProtocol
from vrtool.orm.io.importers.slope_part_importer import SlopePartImporter
from vrtool.orm.models.computation_scenario import ComputationScenario
from vrtool.orm.models.grass_revetment_relation import GrassRevetmentRelation
from vrtool.orm.models.slope_part import SlopePart


class RevetmentImporter(OrmImporterProtocol):
    _slope_part_importer: SlopePartImporter

    def __init__(self) -> None:
        self._slope_part_importer = SlopePartImporter()

    def _get_grass_revetment_relations(
        self, relations: list[GrassRevetmentRelation]
    ) -> list[RelationGrassRevetment]:
        return [
            RelationGrassRevetment(
                relation.year, relation.transition_level, relation.beta
            )
            for relation in relations
        ]

    def _get_slope_parts(self, slope_parts: list[SlopePart]) -> list[SlopePartProtocol]:
        return [self._slope_part_importer.import_orm(part) for part in slope_parts]

    def _is_revetment_data_valid(self, input: RevetmentDataClass):
        actual_transition_level = input.current_transition_level

        maximum_transition_level_relation = max(
            [
                grass_relation.transition_level
                for grass_relation in input.grass_relations
            ]
        )

        return actual_transition_level < maximum_transition_level_relation

    def import_orm(self, orm_model: ComputationScenario) -> MechanismInput:
        if not orm_model:
            raise ValueError(
                f"No valid value given for {ComputationScenario.__name__}."
            )

        slope_parts = self._get_slope_parts(
            orm_model.slope_parts.select().order_by(SlopePart.begin_part)
        )

        grass_relations = self._get_grass_revetment_relations(
            orm_model.grass_revetment_relations.select().order_by(
                GrassRevetmentRelation.year, GrassRevetmentRelation.transition_level
            )
        )

        if not any(grass_relations):
            raise ValueError(
                f"No grass revetment relations for scenario {orm_model.scenario_name}."
            )

        revetment_input = RevetmentDataClass(slope_parts, grass_relations)

        if not self._is_revetment_data_valid(revetment_input):
            raise ValueError(
                f"Actual transition level higher than maximum transition level of grass revetment relations for scenario {orm_model.scenario_name}."
            )

        mechanism_input = MechanismInput("Revetment")
        mechanism_input.input["revetment_input"] = revetment_input

        return mechanism_input
