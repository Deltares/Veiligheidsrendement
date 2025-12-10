import json
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Union

import pytest

from tests import test_data, test_results
from vrtool.common.enums.mechanism_enum import MechanismEnum
from vrtool.common.measure_unit_costs import MeasureUnitCosts
from vrtool.defaults.vrtool_config import VrtoolConfig


class TestVrtoolConfig:
    def test_init_vrtool_config_default_values(self):
        # 1. Define test data.
        _expected_keys = [
            "language",
            "traject",
            "input_directory",
            "t_0",
            "T",
            "excluded_mechanisms",
            "crest_step",
            "berm_step",
            "OI_horizon",
            "BC_stop",
            "max_greedy_iterations",
            "f_cautious",
            "discount_rate",
            "design_methods",
            "unit_costs",
            "externals",
            "input_database_name",
        ]

        # 2. Run test.
        _config = VrtoolConfig()

        # 3. Verify expectations.
        assert isinstance(_config, VrtoolConfig)

        expected_set = set(_expected_keys)
        actual_keys_set = set(asdict(_config).keys())
        _different_entries = expected_set.symmetric_difference(actual_keys_set)
        assert not any(
            _different_entries
        ), "Mismatch between expected entries and retrieved: {}".format(
            ",".join(_different_entries)
        )

        # Verify default values.
        assert _config.language == "EN"
        assert _config.input_directory is None

        assert _config.t_0 == 2025
        assert _config.T == [0, 19, 20, 25, 50, 75, 100]
        assert _config.mechanisms == [
            MechanismEnum.OVERFLOW,
            MechanismEnum.STABILITY_INNER,
            MechanismEnum.PIPING,
            MechanismEnum.REVETMENT,
        ]
        assert _config.crest_step == pytest.approx(0.5)
        assert _config.berm_step == [0, 5, 8, 10, 12, 15, 20, 30]
        assert _config.OI_horizon == 50
        assert _config.BC_stop == pytest.approx(0.1)
        assert _config.max_greedy_iterations == 150
        assert _config.f_cautious == pytest.approx(1.5)
        assert _config.discount_rate == pytest.approx(0.03)
        assert _config.design_methods == ["Veiligheidsrendement", "Doorsnede-eisen"]
        assert isinstance(_config.unit_costs, MeasureUnitCosts)

    def test_export(self, request: pytest.FixtureRequest):
        # 1. Define test data.
        _test_dir = test_results / request.node.name
        if _test_dir.exists():
            shutil.rmtree(_test_dir)

        _test_file = _test_dir / "export_config.json"
        _vrtool_config = VrtoolConfig()
        _vrtool_config.traject = "test_traject"

        # 2. Run test
        _vrtool_config.export(_test_file)

        # 3. Verify expectations
        _expected_data = {"traject": _vrtool_config.traject}
        assert _test_file.exists()
        assert _expected_data == json.loads(_test_file.read_text())

    def test_load(self):
        # 1. Define test data.
        _test_file = test_data / "vrtool_config" / "custom_config.json"
        assert _test_file.exists()

        # 2. Run test.
        _vrtool_config = VrtoolConfig.from_json(_test_file)

        # 3. Verify expectations.
        assert isinstance(_vrtool_config, VrtoolConfig)
        assert _vrtool_config.traject == "MyCustomTraject"

    def test_serialize_vrtool_config(self):
        # 1. Define test data.
        _traject_name = "test_traject"
        _vrtool_config = VrtoolConfig(
            input_directory=Path("input"),
            output_directory=Path("output"),
            externals=Path("externals"),
            traject=_traject_name,
        )

        # 2. Run test
        _json_str = _vrtool_config.serialize()
        _loaded_json = json.loads(_json_str)
        # 3. Verify expectations.
        assert _loaded_json is not None
        assert _loaded_json["input_database_name"] == ""
        assert _loaded_json["input_directory"] == "input"
        assert _loaded_json["output_directory"] == "output"
        assert _loaded_json["externals"] == "externals"
        assert _loaded_json["language"] == "EN"
        assert _loaded_json["traject"] == _traject_name
        assert _loaded_json["t_0"] == 2025
        assert _loaded_json["T"] == [0, 19, 20, 25, 50, 75, 100]
        assert _loaded_json["excluded_mechanisms"] == [
            MechanismEnum.HYDRAULIC_STRUCTURES.name
        ]
        assert _loaded_json["crest_step"] == 0.5
        assert _loaded_json["berm_step"] == [0, 5, 8, 10, 12, 15, 20, 30]
        assert _loaded_json["OI_horizon"] == 50
        assert _loaded_json["BC_stop"] == 0.1
        assert _loaded_json["max_greedy_iterations"] == 150
        assert _loaded_json["f_cautious"] == 1.5
        assert _loaded_json["discount_rate"] == 0.03
        assert _loaded_json["design_methods"] == [
            "Veiligheidsrendement",
            "Doorsnede-eisen",
        ]
        assert _loaded_json["unit_costs"] == {
            "inward_added_volume": 58.76,
            "inward_starting_costs": 252.44,
            "outward_reuse_factor": 0.7,
            "outward_removed_volume": 27.59,
            "outward_reused_volume": 18.59,
            "outward_added_volume": 50.75,
            "outward_compensation_factor": 0.5,
            "house_removal": 500000.0,
            "road_renewal": 1070.0,
            "sheetpile": 690.0,
            "diaphragm_wall": 34190.0,
            "vertical_geotextile": 1700.0,
            "coarse_sand_barrier": 1700.0,
            "anchored_sheetpile": 1100.0,
            "heavescreen": 400.0,
            "remove_block_revetment": 15.66,
            "remove_asphalt_revetment": 33.92,
            "installation_of_blocks": {
                "30.0": 206.89,
                "35.0": 235.93,
                "40.0": 264.06,
                "45.0": 291.16,
                "50.0": 318.26,
            },
        }

    @pytest.mark.parametrize(
        "custom_path",
        [
            pytest.param("just\\a\\path", id="Double slash"),
            pytest.param(r"with\simple\slash", id="Simple slash"),
        ],
    )
    def test_init_with_mapproperty_as_str_sets_to_path(self, custom_path: str):
        # 1. Define test data
        _paths_dict = dict(input_directory=custom_path)

        # 2. Run test
        _vrtool_config = VrtoolConfig(**_paths_dict)

        # 3. Verify expectations.
        _custom_path = Path(custom_path)
        assert _vrtool_config.input_directory == _custom_path

    @pytest.mark.parametrize(
        "none_value",
        [pytest.param("", id="Empty string"), pytest.param(None, id="None")],
    )
    def test_init_with_not_value_returns_none(self, none_value: Union[str, None]):
        # 1. Define test data
        _paths_dict = dict(input_directory=none_value)

        # 2. Run test
        _vrtool_config = VrtoolConfig(**_paths_dict)

        # 3. Verify expectations.
        assert _vrtool_config.input_directory is None

    def test_init_with_path_returns_same(self):
        # 1. Define test data
        _test_path = Path("just\\a\\path")
        _paths_dict = dict(input_directory=_test_path)

        # 2. Run test
        _vrtool_config = VrtoolConfig(**_paths_dict)

        # 3. Verify expectations.
        assert _vrtool_config.input_directory == _test_path

    @pytest.mark.parametrize(
        "input_directory",
        [
            pytest.param(Path(r"X:\any\folder"), id="VALID input directory"),
            pytest.param(None, id="NONE input directory"),
            pytest.param(Path(""), id="EMPTY input directory"),
        ],
    )
    @pytest.mark.parametrize(
        "db_name",
        [
            pytest.param("MyDb.db", id="VALID DB name"),
            pytest.param(None, id="NONE DB name"),
            pytest.param("", id="EMPTY DB name"),
        ],
    )
    def test_input_database_path(self, input_directory: Path, db_name: str):
        # 1. Define test data
        _vrtool_config = VrtoolConfig(
            input_directory=input_directory, input_database_name=db_name
        )

        # 2. Run test
        _test_db_path = _vrtool_config.input_database_path

        # 3. Verify expectations
        if input_directory and db_name:
            _expectation = input_directory.joinpath(db_name)
        else:
            _expectation = None
        assert _test_db_path == _expectation

    _available_mechanisms = [
        MechanismEnum.OVERFLOW,
        MechanismEnum.STABILITY_INNER,
        MechanismEnum.PIPING,
        MechanismEnum.REVETMENT,
        MechanismEnum.HYDRAULIC_STRUCTURES,
    ]

    @pytest.mark.parametrize(
        "excluded_mechanisms, expected",
        [
            pytest.param(
                _available_mechanisms[3:], _available_mechanisms[:3], id="VALID filter"
            ),
            pytest.param([None], _available_mechanisms[:], id="NONE filter"),
        ],
    )
    def test_filter_mechanisms(self, excluded_mechanisms, expected):
        # 1. Define test data
        _vrtool_config = VrtoolConfig(excluded_mechanisms=excluded_mechanisms)

        # 2. Run test
        _mechanisms = _vrtool_config.mechanisms

        # 3. Verify expectations
        assert all(_mech in expected for _mech in _mechanisms)

    @pytest.mark.parametrize(
        "path_value, expected_value",
        [
            pytest.param(
                Path(".", "my_relative_path"),
                test_results / "my_relative_path",
                id="Relative path",
            ),
            pytest.param(
                test_data / "my_absolute_path",
                test_data / "my_absolute_path",
                id="Absolute path",
            ),
            pytest.param(None, None, id="No Path"),
        ],
    )
    def test_relative_paths_to_absolute_given_relative_path(
        self, path_value: Path, expected_value: Path
    ):
        # 1. Define test data.
        _vrtool_config = VrtoolConfig()
        _vrtool_config.input_directory = path_value
        _vrtool_config.externals = path_value

        # 2. Run test.
        _vrtool_config._relative_paths_to_absolute(test_results)

        # 3. Verify expectations.
        assert _vrtool_config.input_directory == expected_value
        assert _vrtool_config.externals == expected_value

    def test_t0_and_t100_included_in_T(self):
        # 1. Define test data.
        _dummy_t = [24, 42]
        _expected_values = [0, 24, 42, 100]

        # 2. Run test.
        _vrtool_config = VrtoolConfig(T=_dummy_t)

        # 3. Verify expectations
        assert _vrtool_config.T == _expected_values

    def test_given_default_vrtool_config_when_validate_succeeds(self):
        # 1. Deifne test data / Run test / verify expectations.
        VrtoolConfig().validate_config()

    @pytest.mark.parametrize(
        "t_values",
        [
            pytest.param([0, 23, 42], id="Without 100"),
            pytest.param([23, 42, 100], id="Without 0"),
            pytest.param([23, 42], id="Without 0 and 100"),
        ],
    )
    def test_given_t_without_required_values_when_validate_raises_exception(
        self, t_values: list[int]
    ):
        # 1. Define test data.
        _vrtool_config = VrtoolConfig()
        _vrtool_config.T = t_values

        # 2. Run test
        with pytest.raises(ValueError) as exc_err:
            _vrtool_config.validate_config()

        # 3. Verify expectations.
        assert (
            str(exc_err.value)
            == "'VrtoolConfig' is niet geldig, het vereist de waarden: 0, 100"
        )
