from dataclasses import asdict

import pytest

from src.defaults.vrtool_config import VrtoolConfig, _load_default_unit_costs


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
            "directory",
            "t_0",
            "T",
            "mechanisms",
            "LE_in_section",
            "crest_step",
            "berm_step",
            "OI_year",
            "OI_horizon",
            "BC_stop",
            "max_greedy_iterations",
            "f_cautious",
            "shelves",
            "reuse_output",
            "beta_or_prob",
            "plot_reliability_in_time",
            "plot_measure_reliability",
            "flip_traject",
            "assessment_plot_years",
            "geometry_plot",
            "beta_cost_settings",
            "unit_costs",
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
        assert _config.directory is None

        assert _config.t_0 == 2025
        assert _config.T == [0, 19, 20, 25, 50, 75, 100]
        assert _config.mechanisms == [
            "Overflow",
            "StabilityInner",
            "Piping",
        ]
        assert not _config.LE_in_section
        assert _config.crest_step == pytest.approx(0.5)
        assert _config.berm_step == [0, 5, 8, 10, 12, 15, 20, 30]
        assert _config.OI_year == 0
        assert _config.OI_horizon == 50
        assert _config.BC_stop == pytest.approx(0.1)
        assert _config.max_greedy_iterations == 150
        assert _config.f_cautious == pytest.approx(1.5)
        assert not _config.shelves
        assert not _config.reuse_output
        assert _config.beta_or_prob == "beta"
        assert not _config.plot_reliability_in_time
        assert not _config.plot_measure_reliability
        assert _config.flip_traject
        assert _config.assessment_plot_years == [0, 20, 50]
        assert not _config.geometry_plot
        assert _config.beta_cost_settings == {"symbols": True, "markersize": 10}
        assert isinstance(_config.unit_costs, dict)
        assert any(_config.unit_costs.items())
