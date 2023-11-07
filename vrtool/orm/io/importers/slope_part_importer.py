from vrtool.failure_mechanisms.revetment.relation_stone_revetment import (
    RelationStoneRevetment,
)
from vrtool.failure_mechanisms.revetment.slope_part import (
    SlopePartBuilder,
    SlopePartProtocol,
    StoneSlopePart,
)
from vrtool.orm.io.importers.orm_importer_protocol import OrmImporterProtocol
from vrtool.orm.models.block_revetment_relation import BlockRevetmentRelation
from vrtool.orm.models.slope_part import SlopePart


class SlopePartImporter(OrmImporterProtocol):
    def import_orm(self, orm_model: SlopePart) -> SlopePartProtocol:
        if not orm_model:
            raise ValueError(f"No valid value given for {SlopePart.__name__}.")
        if orm_model.tan_alpha < 0:
            raise ValueError(
                f"tan_alpha of slope part is smaller than 0 ({orm_model.tan_alpha})."
            )
        if orm_model.tan_alpha == 0:
            raise ValueError(
                f"tan_alpha of slope part is equal to 0. Ensure that berms always have a positive value for tan_alpha."
            )            
        imported_part = SlopePartBuilder.build(
            begin_part=orm_model.begin_part,
            end_part=orm_model.end_part,
            tan_alpha=orm_model.tan_alpha,
            top_layer_type=orm_model.top_layer_type,
            top_layer_thickness=orm_model.top_layer_thickness,
        )

        if isinstance(imported_part, StoneSlopePart):
            imported_part.slope_part_relations = self._get_stone_revetment_relations(
                orm_model.block_revetment_relations.select().order_by(
                    BlockRevetmentRelation.year,
                    BlockRevetmentRelation.top_layer_thickness,
                )
            )

        return imported_part

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
