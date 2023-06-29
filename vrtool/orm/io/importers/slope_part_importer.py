from vrtool.failure_mechanisms.revetment.relation_revetment_protocol import (
    RelationRevetmentProtocol,
)
from vrtool.failure_mechanisms.revetment.relation_stone_revetment import (
    RelationStoneRevetment,
)
from vrtool.failure_mechanisms.revetment.slope_part_protocol import SlopePartProtocol
from vrtool.failure_mechanisms.revetment.slope_part_builder import SlopePartBuilder
from vrtool.failure_mechanisms.revetment.stone_slope_part import StoneSlopePart
from vrtool.orm.io.importers.orm_importer_protocol import OrmImporterProtocol
from vrtool.orm.models.block_revetment_relation import BlockRevetmentRelation
from vrtool.orm.models.slope_part import SlopePart


class SlopePartImporter(OrmImporterProtocol):
    def import_orm(self, orm_model: SlopePart) -> SlopePartProtocol:
        if not orm_model:
            raise ValueError(f"No valid value given for {SlopePart.__name__}.")

        imported_part = SlopePartBuilder.build(
            **{
                "begin_part": orm_model.begin_part,
                "end_part": orm_model.end_part,
                "tan_alpha": orm_model.tan_alpha,
                "top_layer_type": orm_model.top_layer_type,
                "top_layer_thickness": orm_model.top_layer_thickness,
            }
        )

        self._set_slope_part_revetment_relations(imported_part, orm_model)

        return imported_part

    def _set_slope_part_revetment_relations(
        self, imported_slope_part: SlopePartProtocol, orm_model: SlopePart
    ) -> list[RelationRevetmentProtocol]:
        if isinstance(imported_slope_part, StoneSlopePart):
            imported_slope_part.slope_part_relations = (
                self._get_stone_revetment_relations(
                    orm_model.block_revetment_relations.select().order_by(
                        BlockRevetmentRelation.year,
                        BlockRevetmentRelation.top_layer_thickness,
                    )
                )
            )

    def _get_stone_revetment_relations(
        self, relations: list[BlockRevetmentRelation]
    ) -> list[RelationStoneRevetment]:
        return [
            RelationStoneRevetment(
                relation.year,
                relation.top_layer_thickness,
                relation.beta,
            )
            for relation in relations
        ]
