import json
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Union

import pytest

from tests import test_data, test_results
from vrtool.common.enums import MechanismEnum
from vrtool.defaults.vrtool_config import VrtoolConfig, _load_default_unit_costs


class TestVrtoolConfig:
    def test_load_default_unit_costs(self):
        # 1. Define test data.
        _expected_keys = [
            "Inward starting costs",
            "Inward added volume",
            "Outward added volume",
            "Outward reused volume",
            "Outward reuse factor",
            "Outward compensation factor",
            "Outward removed volume",
            "Road renewal",
            "Sheetpile",
            "Diaphragm wall",
            "Vertical Geotextile",
            "House removal",
        ]

        # 2. Run test.
        _unit_costs_data = _load_default_unit_costs()

        # 3. Verify expectations.
        assert isinstance(_unit_costs_data, dict)
        assert not (any(set(_expected_keys) - set(_unit_costs_data.keys())))

    def test_init_vrtool_config_default_values(self):
        # 1. Define test data.
        _expected_keys = [
            "language",
            "traject",
            "input_directory",
            "output_directory",
            "t_0",
            "T",
            "excluded_mechanisms",
            "LE_in_section",
            "crest_step",
            "berm_step",
            "OI_year",
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
        assert not _config.LE_in_section
        assert _config.crest_step == pytest.approx(0.5)
        assert _config.berm_step == [0, 5, 8, 10, 12, 15, 20, 30]
        assert _config.OI_year == 0
        assert _config.OI_horizon == 50
        assert _config.BC_stop == pytest.approx(0.1)
        assert _config.max_greedy_iterations == 150
        assert _config.f_cautious == pytest.approx(1.5)
        assert _config.discount_rate == pytest.approx(0.03)
        assert _config.design_methods == ["Veiligheidsrendement", "Doorsnede-eisen"]
        assert isinstance(_config.unit_costs, dict)
        assert any(_config.unit_costs.items())

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

    @pytest.mark.parametrize(
        "custom_path",
        [
            pytest.param("just\\a\\path", id="Double slash"),
            pytest.param(r"with\simple\slash", id="Simple slash"),
        ],
    )
    def test_init_with_mapproperty_as_str_sets_to_path(self, custom_path: str):
        # 1. Define test data
        _paths_dict = dict(output_directory=custom_path, input_directory=custom_path)

        # 2. Run test
        _vrtool_config = VrtoolConfig(**_paths_dict)

        # 3. Verify expectations.
        _custom_path = Path(custom_path)
        assert _vrtool_config.output_directory == _custom_path
        assert _vrtool_config.input_directory == _custom_path

    @pytest.mark.parametrize(
        "none_value",
        [pytest.param("", id="Empty string"), pytest.param(None, id="None")],
    )
    def test_init_with_not_value_returns_none(self, none_value: Union[str, None]):
        # 1. Define test data
        _paths_dict = dict(output_directory=none_value, input_directory=none_value)

        # 2. Run test
        _vrtool_config = VrtoolConfig(**_paths_dict)

        # 3. Verify expectations.
        assert _vrtool_config.output_directory is None
        assert _vrtool_config.input_directory is None

    def test_init_with_path_returns_same(self):
        # 1. Define test data
        _test_path = Path("just\\a\\path")
        _paths_dict = dict(output_directory=_test_path, input_directory=_test_path)

        # 2. Run test
        _vrtool_config = VrtoolConfig(**_paths_dict)

        # 3. Verify expectations.
        assert _vrtool_config.input_directory == _test_path
        assert _vrtool_config.output_directory == _test_path

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
                Path(".\\my_relative_path"),
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
        _vrtool_config.output_directory = path_value
        _vrtool_config.externals = path_value

        # 2. Run test.
        _vrtool_config._relative_paths_to_absolute(test_results)

        # 3. Verify expectations.
        assert _vrtool_config.input_directory == expected_value
        assert _vrtool_config.output_directory == expected_value
        assert _vrtool_config.externals == expected_value
