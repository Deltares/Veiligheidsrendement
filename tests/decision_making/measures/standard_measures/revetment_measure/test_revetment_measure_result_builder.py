import csv
import itertools
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from tests import get_clean_test_results_dir, test_data, test_results
from tests.failure_mechanisms.revetment.json_files_to_revetment_dataclass_reader import (
    JsonFilesToRevetmentDataClassReader,
)
from vrtool.common.measure_unit_costs import MeasureUnitCosts
from vrtool.decision_making.measures.standard_measures.revetment_measure.revetment_measure_data import (
    RevetmentMeasureData,
)
from vrtool.decision_making.measures.standard_measures.revetment_measure.revetment_measure_result import (
    RevetmentMeasureResult,
)
from vrtool.decision_making.measures.standard_measures.revetment_measure.revetment_measure_result_builder import (
    RevetmentMeasureResultBuilder,
)
from vrtool.failure_mechanisms.revetment.relation_grass_revetment import (
    RelationGrassRevetment,
)
from vrtool.failure_mechanisms.revetment.relation_stone_revetment import (
    RelationStoneRevetment,
)
from vrtool.failure_mechanisms.revetment.revetment_data_class import RevetmentDataClass
from vrtool.failure_mechanisms.revetment.slope_part import (
    GrassSlopePart,
    StoneSlopePart,
)
from vrtool.failure_mechanisms.revetment.slope_part.asphalt_slope_part import (
    AsphaltSlopePart,
)

from vrtool.defaults import default_unit_costs_csv

@dataclass
class JsonFileCase:
    evaluation_year: int
    given_years: list[int]  # Options are 2025, 2100 (or both)
    section_id: int
    crest_height: float
    target_beta: float
    transition_level: float
    section_length: float


def get_json_file_cases(
    section_id: int,
    crest_height: float,
    target_beta: float,
    section_length: float,
    transition_level_vector: list[float],
) -> list[JsonFileCase]:
    _available_years = [2025, 2100]
    return [
        JsonFileCase(
            evaluation_year=_evaluation_year,
            given_years=_available_years,
            section_id=section_id,
            crest_height=crest_height,
            target_beta=target_beta,
            transition_level=_transition_level,
            section_length=section_length,
        )
        for _transition_level, _evaluation_year in itertools.product(
            transition_level_vector, _available_years
        )
    ]


def get_pytest_json_file_cases(
    json_file_case_list: list[JsonFileCase],
) -> list[pytest.param]:
    def _wrap_in_pytest_param(json_file_case: JsonFileCase) -> pytest.param:
        return pytest.param(
            json_file_case,
            id="Section {} year {} target_beta {} transition_level {}".format(
                json_file_case.section_id,
                json_file_case.evaluation_year,
                json_file_case.target_beta,
                json_file_case.transition_level,
            ),
        )

    return list(map(_wrap_in_pytest_param, json_file_case_list))


_section_1_cases = get_json_file_cases(0, 5.87, 3.4029328353853043, 50, [3.99])
_section_2_cases = get_json_file_cases(
    2, 10.27, 5.3342534521987, 766, np.arange(3.76, 10.27, 0.25)
)
_json_file_cases = _section_1_cases + _section_2_cases
_pytest_json_file_cases = get_pytest_json_file_cases(_json_file_cases)


class TestRevetmentMeasureResultBuilder:
    @pytest.mark.parametrize(
        "revetment_data", [pytest.param(None), pytest.param(RevetmentDataClass())]
    )
    def test_get_revetment_measures_collection_no_slope_parts(
        self, revetment_data: RevetmentDataClass
    ):
        # 1. Define test data.
        _crest_height = 4.2
        _target_beta = 0.4
        _transition_level = 2.4
        _evaluation_year = 2023
        _builder = RevetmentMeasureResultBuilder()

        # 2. Run test.
        _data_collection = _builder._get_revetment_measures_collection(
            _crest_height,
            revetment_data,
            _target_beta,
            _transition_level,
            _evaluation_year,
        )

        # 3. Verify expectations.
        assert isinstance(_data_collection, list)
        assert not any(_data_collection)

    def test_get_revetment_measures_collection(self):
        # 1. Define test data.
        _revetment_data = RevetmentDataClass()
        _revetment_data.slope_parts = [
            StoneSlopePart(
                begin_part=-0.27,
                end_part=1.89,
                tan_alpha=0.25064,
                top_layer_thickness=0.2,
                top_layer_type=26.0,
                slope_part_relations=[
                    RelationStoneRevetment(
                        beta=3.6623, top_layer_thickness=0.2021, year=2025
                    ),
                    RelationStoneRevetment(
                        beta=4.7081, top_layer_thickness=0.2469, year=2025
                    ),
                    RelationStoneRevetment(
                        beta=5.3597, top_layer_thickness=0.2538, year=2025
                    ),
                    RelationStoneRevetment(
                        beta=5.5732, top_layer_thickness=0.2640, year=2025
                    ),
                    RelationStoneRevetment(
                        beta=3.6622, top_layer_thickness=0.2021, year=2100
                    ),
                    RelationStoneRevetment(
                        beta=4.7081, top_layer_thickness=0.2539, year=2100
                    ),
                    RelationStoneRevetment(
                        beta=5.3597, top_layer_thickness=0.2600, year=2100
                    ),
                    RelationStoneRevetment(
                        beta=5.5732, top_layer_thickness=0.2699, year=2100
                    ),
                ],
            ),
            StoneSlopePart(
                begin_part=1.89,
                end_part=3.86,
                tan_alpha=0.34377,
                top_layer_thickness=0.275,
                top_layer_type=26.1,
                slope_part_relations=[
                    RelationStoneRevetment(
                        beta=3.6623, top_layer_thickness=0.2195, year=2025
                    ),
                    RelationStoneRevetment(
                        beta=4.7081, top_layer_thickness=0.2976, year=2025
                    ),
                    RelationStoneRevetment(
                        beta=5.3597, top_layer_thickness=0.3386, year=2025
                    ),
                    RelationStoneRevetment(
                        beta=5.5732, top_layer_thickness=0.3514, year=2025
                    ),
                    RelationStoneRevetment(
                        beta=3.6622, top_layer_thickness=0.2195, year=2100
                    ),
                    RelationStoneRevetment(
                        beta=4.7081, top_layer_thickness=0.2976, year=2100
                    ),
                    RelationStoneRevetment(
                        beta=5.3597, top_layer_thickness=0.3526, year=2100
                    ),
                    RelationStoneRevetment(
                        beta=5.5732, top_layer_thickness=0.3667, year=2100
                    ),
                ],
            ),
            AsphaltSlopePart(
                begin_part=3.86,
                end_part=3.98,
                tan_alpha=0.03709,
                top_layer_thickness=20.0,
                top_layer_type=5.01,
            ),
            GrassSlopePart(
                begin_part=3.98,
                end_part=3.99,
                tan_alpha=0.01138,
                top_layer_thickness=None,
                top_layer_type=20.0,
            ),
            GrassSlopePart(
                begin_part=3.99,
                end_part=5.87,
                tan_alpha=0.36573,
                top_layer_thickness=None,
                top_layer_type=20.0,
            ),
        ]
        _revetment_data.grass_relations = [
            RelationGrassRevetment(beta=4.90, transition_level=4.0, year=2025),
            RelationGrassRevetment(beta=4.90, transition_level=4.25, year=2025),
            RelationGrassRevetment(beta=4.90, transition_level=4.5, year=2025),
            RelationGrassRevetment(beta=4.94, transition_level=5.0, year=2025),
            RelationGrassRevetment(beta=5.03, transition_level=5.5, year=2025),
            RelationGrassRevetment(beta=4.74, transition_level=4.0, year=2100),
            RelationGrassRevetment(beta=4.74, transition_level=4.25, year=2100),
            RelationGrassRevetment(beta=4.75, transition_level=4.5, year=2100),
            RelationGrassRevetment(beta=4.76, transition_level=5.0, year=2100),
            RelationGrassRevetment(beta=4.80, transition_level=5.5, year=2100),
        ]
        _crest_height = 4.2
        _target_beta = 0.4
        _transition_level = 2.4
        _evaluation_year = 2025
        _builder = RevetmentMeasureResultBuilder()

        # 2. Run test.
        _data_collection = _builder._get_revetment_measures_collection(
            _crest_height,
            _revetment_data,
            _target_beta,
            _transition_level,
            _evaluation_year,
        )

        # 3. Verify expectations.
        _expected_matrix = [
            RevetmentMeasureData(
                begin_part=-0.27,
                beta_block_revetment=3.6132781250000003,
                beta_grass_revetment=float("nan"),
                end_part=1.89,
                previous_top_layer_type=26.0,
                reinforce=False,
                tan_alpha=0.25064,
                top_layer_thickness=0.2,
                top_layer_type=26.0,
            ),
            RevetmentMeasureData(
                begin_part=1.89,
                beta_block_revetment=float("nan"),
                beta_grass_revetment=float("nan"),
                end_part=2.4,
                previous_top_layer_type=26.1,
                reinforce=False,
                tan_alpha=0.34377,
                top_layer_thickness=float("nan"),
                top_layer_type=26.1,
            ),
            RevetmentMeasureData(
                begin_part=2.4,
                beta_block_revetment=float("nan"),
                beta_grass_revetment=4.9,
                end_part=3.86,
                previous_top_layer_type=26.1,
                reinforce=True,
                tan_alpha=0.34377,
                top_layer_thickness=float("nan"),
                top_layer_type=20.0,
            ),
            RevetmentMeasureData(
                begin_part=3.86,
                beta_block_revetment=float("nan"),
                beta_grass_revetment=4.9,
                end_part=3.98,
                previous_top_layer_type=5.01,
                reinforce=True,
                tan_alpha=0.3709,
                top_layer_thickness=float("nan"),
                top_layer_type=20.0,
            ),
            RevetmentMeasureData(
                begin_part=3.98,
                beta_block_revetment=float("nan"),
                beta_grass_revetment=float("nan"),
                end_part=3.99,
                previous_top_layer_type=20,
                reinforce=True,
                tan_alpha=0.01138,
                top_layer_thickness=float("nan"),
                top_layer_type=20.0,
            ),
            RevetmentMeasureData(
                begin_part=3.99,
                beta_block_revetment=float("nan"),
                beta_grass_revetment=4.9,
                end_part=5.87,
                previous_top_layer_type=20.0,
                reinforce=True,
                tan_alpha=0.36573,
                top_layer_thickness=float("nan"),
                top_layer_type=20.0,
            ),
        ]
        assert isinstance(_data_collection, list)
        assert len(_data_collection) == 6
        assert all(
            isinstance(_data, RevetmentMeasureData) for _data in _data_collection
        )
        assert all(
            sorted(_expected_matrix[i].__dict__) == sorted(_data_collection[i].__dict__)
            for i in range(0, 6)
        )

    @pytest.mark.parametrize(
        "json_file_case",
        _pytest_json_file_cases,
    )
    def test_get_revetment_measures_collection_from_json_files(
        self, json_file_case: JsonFileCase, request: pytest.FixtureRequest
    ):
        # Note: This test is meant so that results can be verified in TC.
        # 1. Define test data.
        _builder = RevetmentMeasureResultBuilder()
        _revetment_data = JsonFilesToRevetmentDataClassReader().get_revetment_input(
            json_file_case.given_years, json_file_case.section_id
        )

        # 2. Run test.
        _results = _builder._get_revetment_measures_collection(
            json_file_case.crest_height,
            _revetment_data,
            json_file_case.target_beta,
            json_file_case.transition_level,
            json_file_case.evaluation_year,
        )

        # 3. Verify expectations.
        assert isinstance(_results, list)
        assert all(isinstance(r, RevetmentMeasureData) for r in _results)

        # 4. Output results.
        def measure_to_dict(measure: RevetmentMeasureData) -> dict:
            measure.cost = measure.get_total_cost(json_file_case.section_length, MeasureUnitCosts.from_csv_file(default_unit_costs_csv))
            return measure.__dict__

        self._output_to_csv(
            self._get_testcase_output_filepath(request),
            (list(map(measure_to_dict, _results))),
        )

    def test_build_and_output_collection_from_json_files(
        self, request: pytest.FixtureRequest
    ):
        # Note: This test is meant so that results can be verified in TC.
        # 1. Define test data.
        _builder = RevetmentMeasureResultBuilder()
        _json_reader = JsonFilesToRevetmentDataClassReader()
        _unit_costs = MeasureUnitCosts.from_csv_file(default_unit_costs_csv)
        # 2. Run test.
        _results = []
        for _case in _json_file_cases:
            _result = _builder.build(
                _case.crest_height,
                _case.section_length,
                _json_reader.get_revetment_input(_case.given_years, _case.section_id),
                _case.target_beta,
                _case.transition_level,
                _case.evaluation_year,
                _unit_costs,
            )
            assert isinstance(_result, RevetmentMeasureResult)
            _results.append({"section_id": _case.section_id} | _result.__dict__)

        # 3. Verify expectations.
        assert any(_results)

        # 4. Output results.
        _output_file = get_clean_test_results_dir(request).joinpath(
            "measures_matrix.csv"
        )
        self._output_to_csv(_output_file, _results)
        assert _output_file.exists()
        assert len(_output_file.read_text().splitlines()) == len(_results) + 1

    def test_compare_revetment_measure_results_cost(self):
        # 1. Define test data.
        _builder = RevetmentMeasureResultBuilder()
        _json_reader = JsonFilesToRevetmentDataClassReader()
        _test_data = test_data.joinpath(
            "revetment_measure_results", "matrix_results.csv"
        )
        assert _test_data.exists()

        # 2. Run test.
        _calculated_costs = []
        for _comparable_case in _section_2_cases:
            if _comparable_case.evaluation_year != 2025:
                # At the moment we only compare the costs for the first year, so no need to generate unrequired results.
                continue
            _result = _builder.build(
                _comparable_case.crest_height,
                _comparable_case.section_length,
                _json_reader.get_revetment_input(
                    _comparable_case.given_years, _comparable_case.section_id
                ),
                _comparable_case.target_beta,
                _comparable_case.transition_level,
                _comparable_case.evaluation_year,
            )

            # 3. Verify expectations.
            assert isinstance(_result, RevetmentMeasureResult)
            _calculated_costs.append(_result.cost)

        # Compare results.
        _expected_results = np.genfromtxt(_test_data, delimiter=",")
        assert np.allclose(_expected_results[1:], np.array(_calculated_costs))

    def _output_to_csv(self, output_file: Path, csv_dicts: list[dict]):
        _header = list(csv_dicts[0].keys())
        with open(
            output_file, "w", newline=""
        ) as f:  # You will need 'wb' mode in Python 2.x
            w = csv.DictWriter(f, _header)
            w.writeheader()
            w.writerows(csv_dicts)

    def _get_testcase_output_filepath(self, request: pytest.FixtureRequest) -> Path:
        _output_file_dir = request.node.name.split("[")[0].strip().lower()
        _output_file_name = (
            request.node.name.split("[")[-1]
            .split("]")[0]
            .strip()
            .lower()
            .replace(" ", "_")
        )
        _output_file = test_results.joinpath(_output_file_dir).joinpath(
            _output_file_name + ".csv"
        )
        _output_file.parent.mkdir(parents=True, exist_ok=True)
        _output_file.unlink(missing_ok=True)

        return _output_file
