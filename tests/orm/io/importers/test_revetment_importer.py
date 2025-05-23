from typing import Callable, Iterator

import pytest
from peewee import SqliteDatabase, _savepoint

from vrtool.common.enums.mechanism_enum import MechanismEnum
from vrtool.failure_mechanisms.mechanism_input import MechanismInput
from vrtool.failure_mechanisms.revetment.relation_grass_revetment import (
    RelationGrassRevetment,
)
from vrtool.failure_mechanisms.revetment.revetment_data_class import RevetmentDataClass
from vrtool.failure_mechanisms.revetment.slope_part import SlopePartProtocol
from vrtool.orm.io.importers.orm_importer_protocol import OrmImporterProtocol
from vrtool.orm.io.importers.revetment_importer import RevetmentImporter
from vrtool.orm.models.block_revetment_relation import BlockRevetmentRelation
from vrtool.orm.models.computation_scenario import ComputationScenario
from vrtool.orm.models.grass_revetment_relation import GrassRevetmentRelation
from vrtool.orm.models.slope_part import SlopePart

general_slope_parts = [
    {
        "begin_part": -0.27,
        "end_part": 1.89,
        "top_layer_type": 26.1,
        "top_layer_thickness": 0.2,
        "tan_alpha": 0.25064,
    },
    {
        "begin_part": 1.89,
        "end_part": 3.86,
        "top_layer_type": 5,
        "top_layer_thickness": 0.275,
        "tan_alpha": 0.34377,
    },
    {
        "begin_part": 3.86,
        "end_part": 3.98,
        "top_layer_type": 20,
        "top_layer_thickness": None,
        "tan_alpha": 0.3709,
    },
]


class TestRevetmentImporter:
    @pytest.fixture(name="computatio_scenario_and_transaction")
    def _get_basic_scenario_fixture(
        self,
        empty_db_context: SqliteDatabase,
        get_basic_computation_scenario: Callable[[], ComputationScenario],
    ) -> Iterator[tuple[ComputationScenario, _savepoint]]:
        with empty_db_context.atomic() as transaction:
            computation_scenario = get_basic_computation_scenario()

            yield (computation_scenario, transaction)

    def test_initialize_revetment_importer(self):
        _importer = RevetmentImporter()
        assert isinstance(_importer, RevetmentImporter)
        assert isinstance(_importer, OrmImporterProtocol)

    def test_import_revetment_basic_scenario(
        self,
        computatio_scenario_and_transaction: tuple[ComputationScenario, _savepoint],
    ):
        # Setup
        grass_relations = [
            {"year": 2025, "transition_level": 4, "beta": 4.90},
            {"year": 2025, "transition_level": 4.25, "beta": 4.95},
            {"year": 2025, "transition_level": 4.5, "beta": 5.03},
        ]

        slope_parts = [
            {
                "begin_part": -0.27,
                "end_part": 1.89,
                "top_layer_type": 26.1,
                "top_layer_thickness": 0.2,
                "tan_alpha": 0.25064,
            },
            {
                "begin_part": 1.89,
                "end_part": 3.86,
                "top_layer_type": 5,
                "top_layer_thickness": 0.275,
                "tan_alpha": 0.34377,
            },
            {
                "begin_part": 3.86,
                "end_part": 3.98,
                "top_layer_type": 20,
                "top_layer_thickness": None,
                "tan_alpha": 0.3709,
            },
        ]

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

        (computation_scenario, transaction) = computatio_scenario_and_transaction
        database_grass_relations = [
            dict(computation_scenario=computation_scenario) | relation
            for relation in grass_relations
        ]
        GrassRevetmentRelation.insert_many(database_grass_relations).execute()

        database_slope_parts = [
            dict(computation_scenario=computation_scenario) | slope_part
            for slope_part in slope_parts
        ]
        created_slope_parts = []
        for database_slope_part in database_slope_parts:
            created_slope_parts.append(SlopePart.create(**database_slope_part))

        database_slope_part_relations = [
            dict(slope_part=created_slope_parts[0]) | stone_relation
            for stone_relation in stone_relations[:2]
        ]
        database_slope_part_relations += [
            dict(slope_part=created_slope_parts[1]) | stone_relation
            for stone_relation in stone_relations[2:]
        ]
        BlockRevetmentRelation.insert_many(database_slope_part_relations).execute()
        transaction.commit

        importer = RevetmentImporter()

        # Call
        _mechanism_input = importer.import_orm(computation_scenario)

        assert isinstance(_mechanism_input, MechanismInput)
        assert _mechanism_input.mechanism == MechanismEnum.REVETMENT
        assert len(_mechanism_input.input) == 1

        revetment_input = _mechanism_input.input["revetment_input"]
        assert isinstance(revetment_input, RevetmentDataClass)
        self._assert_slope_parts(revetment_input.slope_parts, slope_parts)
        self._assert_grass_revetment_relations(
            revetment_input.grass_relations, grass_relations
        )

    def test_import_revetment_with_unsorted_slope_parts_returns_input_with_sorted_parts(
        self,
        computatio_scenario_and_transaction: tuple[ComputationScenario, _savepoint],
    ):
        # Setup
        slope_parts = [
            {
                "begin_part": -0.27,
                "end_part": 1.89,
                "top_layer_type": 26.1,
                "top_layer_thickness": 0.2,
                "tan_alpha": 0.25064,
            },
            {
                "begin_part": 3.86,
                "end_part": 3.98,
                "top_layer_type": 20,
                "top_layer_thickness": None,
                "tan_alpha": 0.3709,
            },
            {
                "begin_part": 1.89,
                "end_part": 3.86,
                "top_layer_type": 5,
                "top_layer_thickness": 0.275,
                "tan_alpha": 0.34377,
            },
        ]

        (computation_scenario, transaction) = computatio_scenario_and_transaction
        database_slope_parts = [
            dict(computation_scenario=computation_scenario) | slope_part
            for slope_part in slope_parts
        ]
        SlopePart.insert_many(database_slope_parts).execute()

        GrassRevetmentRelation.insert(
            computation_scenario=computation_scenario,
            year=2025,
            transition_level=4,
            beta=4.90,
        ).execute()

        transaction.commit

        importer = RevetmentImporter()

        # Call
        _mechanism_input = importer.import_orm(computation_scenario)

        # Assert
        revetment_input = _mechanism_input.input["revetment_input"]
        self._assert_slope_parts(revetment_input.slope_parts, slope_parts)

    def test_import_revetment_with_unsorted_grass_relations_returns_input_with_sorted_relations(
        self,
        computatio_scenario_and_transaction: tuple[ComputationScenario, _savepoint],
    ):
        # Setup
        grass_relations = [
            {"year": 2025, "transition_level": 4.25, "beta": 5.02},
            {"year": 2025, "transition_level": 4, "beta": 5.01},
            {"year": 2100, "transition_level": 4.5, "beta": 4.78},
            {"year": 2100, "transition_level": 4, "beta": 4.75},
            {"year": 2025, "transition_level": 4.5, "beta": 5.03},
            {"year": 2100, "transition_level": 4.25, "beta": 4.77},
        ]

        (computation_scenario, transaction) = computatio_scenario_and_transaction
        database_grass_relations = [
            dict(computation_scenario=computation_scenario) | relation
            for relation in grass_relations
        ]
        GrassRevetmentRelation.insert_many(database_grass_relations).execute()

        database_slope_parts = [
            dict(computation_scenario=computation_scenario) | slope_part
            for slope_part in general_slope_parts
        ]
        SlopePart.insert_many(database_slope_parts).execute()

        transaction.commit

        importer = RevetmentImporter()

        # Call
        _mechanism_input = importer.import_orm(computation_scenario)

        # Assert
        revetment_input = _mechanism_input.input["revetment_input"]
        self._assert_grass_revetment_relations(
            revetment_input.grass_relations, grass_relations
        )

    def test_import_revetment_without_grass_transition_levels_raises_value_error(
        self,
        computatio_scenario_and_transaction: tuple[ComputationScenario, _savepoint],
    ):
        # Setup
        (computation_scenario, transaction) = computatio_scenario_and_transaction
        database_slope_parts = [
            dict(computation_scenario=computation_scenario) | slope_part
            for slope_part in general_slope_parts
        ]
        SlopePart.insert_many(database_slope_parts).execute()

        transaction.commit

        importer = RevetmentImporter()

        # Call
        with pytest.raises(ValueError) as value_error:
            importer.import_orm(computation_scenario)

        # Assert
        _expected_mssg = f"No grass revetment relations for scenario {computation_scenario.scenario_name}."
        assert str(value_error.value) == _expected_mssg

    def test_import_revetment_with_transition_level_larger_than_grass_transition_level_raises_value_error(
        self,
        computatio_scenario_and_transaction: tuple[ComputationScenario, _savepoint],
    ):
        # Setup
        grass_relations = [
            {"year": 2025, "transition_level": 2, "beta": 4.90},
        ]

        (computation_scenario, transaction) = computatio_scenario_and_transaction
        database_grass_relations = [
            dict(computation_scenario=computation_scenario) | relation
            for relation in grass_relations
        ]
        GrassRevetmentRelation.insert_many(database_grass_relations).execute()

        database_slope_parts = [
            dict(computation_scenario=computation_scenario) | slope_part
            for slope_part in general_slope_parts
        ]
        SlopePart.insert_many(database_slope_parts).execute()

        transaction.commit

        importer = RevetmentImporter()

        # Call
        with pytest.raises(ValueError) as value_error:
            importer.import_orm(computation_scenario)

        # Assert
        _expected_message = f"Actual transition level higher than maximum transition level of grass revetment relations for scenario {computation_scenario.scenario_name}."
        assert str(value_error.value) == _expected_message

    def _assert_slope_parts(
        self,
        actual_slope_parts: list[SlopePartProtocol],
        expected_slope_parts: dict[str, float],
    ):
        assert len(actual_slope_parts) == len(expected_slope_parts)

        sorted_expected_slope_parts = sorted(
            expected_slope_parts,
            key=lambda slope_part: slope_part["begin_part"],
        )

        for index, expected_slope_part in enumerate(sorted_expected_slope_parts):
            actual_slope_part = actual_slope_parts[index]
            assert actual_slope_part.begin_part == expected_slope_part["begin_part"]
            assert actual_slope_part.end_part == expected_slope_part["end_part"]
            assert actual_slope_part.tan_alpha == expected_slope_part["tan_alpha"]
            assert (
                actual_slope_part.top_layer_type
                == expected_slope_part["top_layer_type"]
            )
            assert (
                actual_slope_part.top_layer_thickness
                == expected_slope_part["top_layer_thickness"]
            )

    def _assert_grass_revetment_relations(
        self,
        actual_grass_revetment_relations: list[RelationGrassRevetment],
        expected_grass_revetment_relation: dict[str, float],
    ):
        assert len(actual_grass_revetment_relations) == len(
            expected_grass_revetment_relation
        )

        sorted_expected_relations = sorted(
            expected_grass_revetment_relation,
            key=lambda relation: (relation["year"], relation["transition_level"]),
        )

        for index, expected_relation in enumerate(sorted_expected_relations):
            actual_grass_revetment_relation = actual_grass_revetment_relations[index]
            assert actual_grass_revetment_relation.year == expected_relation["year"]
            assert (
                actual_grass_revetment_relation.transition_level
                == expected_relation["transition_level"]
            )
            assert actual_grass_revetment_relation.beta == expected_relation["beta"]
