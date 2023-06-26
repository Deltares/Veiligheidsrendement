from vrtool.failure_mechanisms.mechanism_input import MechanismInput
from vrtool.failure_mechanisms.revetment.relation_grass_revetment import (
    RelationGrassRevetment,
)
from vrtool.failure_mechanisms.revetment.relation_stone_revetment import (
    RelationStoneRevetment,
)
from vrtool.failure_mechanisms.revetment.revetment_data_class import RevetmentDataClass
from vrtool.failure_mechanisms.revetment.slope_part import SlopePart
from vrtool.orm.io.importers.orm_importer_protocol import OrmImporterProtocol
from vrtool.orm.models.block_revetment_relation import BlockRevetmentRelation
from vrtool.orm.models.computation_scenario import ComputationScenario
from vrtool.orm.models.grass_revetment_relation import GrassRevetmentRelation
from vrtool.orm.models.slope_part import SlopePart as ORMSlopePart


class RevetmentImporter(OrmImporterProtocol):
    def _get_grass_revetment_relations(
        self, relations: list[GrassRevetmentRelation]
    ) -> list[RelationGrassRevetment]:
        return [
            RelationGrassRevetment(
                relation.year, relation.transition_level, relation.beta
            )
            for relation in relations
        ]

    def _get_stone_revetment_relations(
        self, relations: list[BlockRevetmentRelation]
    ) -> list[RelationStoneRevetment]:
        return [
            RelationStoneRevetment(
                relation.slope_part.id,
                relation.year,
                relation.top_layer_thickness,
                relation.beta,
            )
            for relation in relations
        ]

    def _get_slope_parts(self, slope_parts: list[ORMSlopePart]) -> list[SlopePart]:
        # TODO: slope part ID is missing
        return [
            SlopePart(
                part.begin_part,
                part.end_part,
                part.tan_alpha,
                part.top_layer_type,
                part.top_layer_thickness,
            )
            for part in slope_parts
        ]

    def import_orm(self, orm_model: ComputationScenario) -> MechanismInput:
        if not orm_model:
            raise ValueError(
                f"No valid value given for {ComputationScenario.__name__}."
            )

        slope_parts = self._get_slope_parts(
            orm_model.slope_parts.select().order_by(ORMSlopePart.begin_part)
        )
        grass_relations = self._get_grass_revetment_relations(
            orm_model.grass_revetment_relations.select().order_by(
                GrassRevetmentRelation.year, GrassRevetmentRelation.transition_level
            )
        )
        stone_relations = self._get_stone_revetment_relations(
            BlockRevetmentRelation.select()
            .join(ORMSlopePart, on=BlockRevetmentRelation.slope_part)
            .join(ComputationScenario, on=ORMSlopePart.computation_scenario)
            .where(ComputationScenario.id == orm_model.get_id())
            .order_by(
                BlockRevetmentRelation.year, BlockRevetmentRelation.top_layer_thickness
            )
        )

        revetment_input = RevetmentDataClass(
            slope_parts, grass_relations, stone_relations
        )

        mechanism_input = MechanismInput("Revetment")
        mechanism_input.input["revetment_input"] = revetment_input

        return mechanism_input
