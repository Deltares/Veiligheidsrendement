from typing import Type
from peewee import SqliteDatabase
import pytest
from vrtool.failure_mechanisms.revetment.relation_stone_revetment import (
    RelationStoneRevetment,
)

from vrtool.failure_mechanisms.revetment.slope_part import SlopePartProtocol
from vrtool.orm.io.importers.orm_importer_protocol import OrmImporterProtocol
from vrtool.orm.io.importers.slope_part_importer import SlopePartImporter
from tests.orm import empty_db_fixture, get_basic_computation_scenario
from vrtool.orm.models.block_revetment_relation import BlockRevetmentRelation
from vrtool.orm.models.slope_part import SlopePart

from vrtool.failure_mechanisms.revetment.asphalt_slope_part import AsphaltSlopePart
from vrtool.failure_mechanisms.revetment.slope_part import SlopePartProtocol
from vrtool.failure_mechanisms.revetment.grass_slope_part import GrassSlopePart
from vrtool.failure_mechanisms.revetment.stone_slope_part import StoneSlopePart


stone_relations = [
    {
        "slope_part_id": 1,
        "year": 2025,
        "top_layer_thickness": 0.202,
        "beta": 3.662,
    },
    {
        "slope_part_id": 1,
        "year": 2100,
        "top_layer_thickness": 0.202,
        "beta": 4.71,
    },
    {
        "slope_part_id": 1,
        "year": 2025,
        "top_layer_thickness": 0.25,
        "beta": 3.562,
    },
    {
        "slope_part_id": 1,
        "year": 2100,
        "top_layer_thickness": 0.25,
        "beta": 5.71,
    },
]


class TestSlopePartImporter:
    def _add_slope_part_id(self, source: list[dict], slope_part_id: int) -> None:
        for item in source:
            item["slope_part_id"] = slope_part_id

    def test_initialize_revetment_importer(self):
        _importer = SlopePartImporter()
        assert isinstance(_importer, SlopePartImporter)
        assert isinstance(_importer, OrmImporterProtocol)

    def test_import_orm_without_model_raises_value_error(self):
        # Setup
        _importer = SlopePartImporter()
        _expected_message = "No valid value given for SlopePart."

        # Call
        with pytest.raises(ValueError) as value_error:
            _importer.import_orm(None)

        # Assert
        assert str(value_error.value) == _expected_message

    @pytest.mark.parametrize(
        "top_layer_type, revetment_type",
        [
            pytest.param(5.0, AsphaltSlopePart, id="Asphalt"),
            pytest.param(20.0, GrassSlopePart, id="Grass"),
        ],
    )
    def test_non_stone_slope_part_returns_expected_slope_part(
        self,
        top_layer_type: float,
        revetment_type: Type[SlopePartProtocol],
        empty_db_fixture: SqliteDatabase,
    ):
        # Setup
        with empty_db_fixture.atomic() as transaction:
            computation_scenario = get_basic_computation_scenario()

            slope_part = SlopePart.create(
                computation_scenario=computation_scenario,
                begin_part=-0.27,
                end_part=1.89,
                top_layer_type=top_layer_type,
                top_layer_thickness=0.2,
                tan_alpha=0.25064,
            )

            self._add_slope_part_id(stone_relations, slope_part.id)
            BlockRevetmentRelation.insert_many(stone_relations).execute()

            transaction.commit

        importer = SlopePartImporter()

        # Call
        imported_part = importer.import_orm(slope_part)

        # Assert
        assert isinstance(imported_part, revetment_type)

        self._assert_slope_parts(imported_part, slope_part)
        assert not any(imported_part.slope_part_relations)

    @pytest.mark.parametrize(
        "top_layer_type",
        [
            pytest.param(26.0, id="Stone slope part type (lower)"),
            pytest.param(27.0, id="Stone slope part type"),
            pytest.param(27.9, id="Stone slope part type (upper)"),
        ],
    )
    def test_stone_slope_part_returns_expected_slope_part(
        self, top_layer_type: float, empty_db_fixture: SqliteDatabase
    ):
        # Setup
        with empty_db_fixture.atomic() as transaction:
            computation_scenario = get_basic_computation_scenario()

            slope_part = SlopePart.create(
                computation_scenario=computation_scenario,
                begin_part=-0.27,
                end_part=1.89,
                top_layer_type=top_layer_type,
                top_layer_thickness=0.2,
                tan_alpha=0.25064,
            )

            self._add_slope_part_id(stone_relations, slope_part.id)
            BlockRevetmentRelation.insert_many(stone_relations).execute()

            transaction.commit

        importer = SlopePartImporter()

        # Call
        imported_part = importer.import_orm(slope_part)

        # Assert
        assert isinstance(imported_part, StoneSlopePart)

        self._assert_slope_parts(imported_part, slope_part)

        imported_slope_part_relations = imported_part.slope_part_relations
        assert all(
            [
                isinstance(relation, RelationStoneRevetment)
                for relation in imported_slope_part_relations
            ]
        )
        self._assert_stone_revetment_relations(
            imported_slope_part_relations, stone_relations
        )

    def _assert_slope_parts(
        self,
        actual_slope_part: SlopePartProtocol,
        expected_slope_part: SlopePart,
    ):
        assert actual_slope_part.begin_part == expected_slope_part.begin_part
        assert actual_slope_part.end_part == expected_slope_part.end_part
        assert actual_slope_part.tan_alpha == expected_slope_part.tan_alpha
        assert actual_slope_part.top_layer_type == expected_slope_part.top_layer_type
        assert (
            actual_slope_part.top_layer_thickness
            == expected_slope_part.top_layer_thickness
        )

    def _assert_stone_revetment_relations(
        self,
        actual_stone_revetment_relations: list[RelationStoneRevetment],
        expected_stone_revetment_relation: dict[str, float],
    ):
        assert len(actual_stone_revetment_relations) == len(
            expected_stone_revetment_relation
        )

        sorted_expected_relations = sorted(
            expected_stone_revetment_relation,
            key=lambda relation: (relation["year"], relation["top_layer_thickness"]),
        )

        for index, expected_relation in enumerate(sorted_expected_relations):
            actual_stone_revetment_relation = actual_stone_revetment_relations[index]
            assert actual_stone_revetment_relation.year == expected_relation["year"]
            assert (
                actual_stone_revetment_relation.top_layer_thickness
                == expected_relation["top_layer_thickness"]
            )
            assert actual_stone_revetment_relation.beta == expected_relation["beta"]
