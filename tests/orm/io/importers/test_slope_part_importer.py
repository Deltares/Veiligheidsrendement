from typing import Type

import pytest
from peewee import SqliteDatabase

from tests.orm import empty_db_fixture, get_basic_computation_scenario
from vrtool.failure_mechanisms.revetment.slope_part import (
    AsphaltSlopePart,
    GrassSlopePart,
    SlopePartProtocol,
    StoneSlopePart,
)
from vrtool.failure_mechanisms.revetment.relation_stone_revetment import (
    RelationStoneRevetment,
)
from vrtool.orm.io.importers.orm_importer_protocol import OrmImporterProtocol
from vrtool.orm.io.importers.slope_part_importer import SlopePartImporter
from vrtool.orm.models.block_revetment_relation import BlockRevetmentRelation
from vrtool.orm.models.slope_part import SlopePart

stone_relations = [
    {
        "year": 2025,
        "top_layer_thickness": 0.202,
        "beta": 3.662,
    },
    {
        "year": 2100,
        "top_layer_thickness": 0.202,
        "beta": 4.71,
    },
    {
        "year": 2025,
        "top_layer_thickness": 0.25,
        "beta": 3.562,
    },
    {
        "year": 2100,
        "top_layer_thickness": 0.25,
        "beta": 5.71,
    },
]


class TestSlopePartImporter:
    @pytest.fixture
    def get_slope_part_fixture(
        self, request: pytest.FixtureRequest, empty_db_fixture: SqliteDatabase
    ):
        with empty_db_fixture.atomic() as transaction:
            computation_scenario = get_basic_computation_scenario()

            slope_part = SlopePart.create(
                computation_scenario=computation_scenario,
                begin_part=-0.27,
                end_part=1.89,
                top_layer_type=request.param,
                top_layer_thickness=0.2,
                tan_alpha=0.25064,
            )

            for relation in stone_relations:
                BlockRevetmentRelation.create(
                    **(relation | dict(slope_part=slope_part))
                )

            transaction.commit

            yield slope_part

    def test_initialize_slope_part_importer(self):
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
        "get_slope_part_fixture, revetment_type",
        [
            pytest.param(5.0, AsphaltSlopePart, id="Asphalt"),
            pytest.param(20.0, GrassSlopePart, id="Grass"),
        ],
        indirect=["get_slope_part_fixture"],
    )
    def test_non_stone_slope_part_returns_expected_slope_part(
        self, get_slope_part_fixture: SlopePart, revetment_type: Type
    ):
        # Setup
        slope_part = get_slope_part_fixture
        importer = SlopePartImporter()

        # Call
        imported_part = importer.import_orm(slope_part)

        # Assert
        assert isinstance(imported_part, revetment_type)

        self._assert_slope_parts(imported_part, slope_part)
        assert not any(imported_part.slope_part_relations)

    @pytest.mark.parametrize(
        "get_slope_part_fixture",
        [
            pytest.param(26.0, id="Stone slope part type (lower)"),
            pytest.param(27.0, id="Stone slope part type"),
            pytest.param(27.9, id="Stone slope part type (upper)"),
        ],
        indirect=["get_slope_part_fixture"],
    )
    def test_stone_slope_part_returns_expected_slope_part(
        self, get_slope_part_fixture: SlopePart
    ):
        # Setup
        slope_part = get_slope_part_fixture
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

        expected_stone_revetment_relation = (
            slope_part.block_revetment_relations.select().order_by(
                BlockRevetmentRelation.year,
                BlockRevetmentRelation.top_layer_thickness,
            )
        )

        assert len(imported_slope_part_relations) == len(
            expected_stone_revetment_relation
        )

        for index, expected_relation in enumerate(expected_stone_revetment_relation):
            actual_stone_revetment_relation = imported_slope_part_relations[index]
            assert actual_stone_revetment_relation.year == expected_relation.year
            assert (
                actual_stone_revetment_relation.top_layer_thickness
                == expected_relation.top_layer_thickness
            )
            assert actual_stone_revetment_relation.beta == expected_relation.beta

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
