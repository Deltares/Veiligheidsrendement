from dataclasses import asdict

import pytest

from src.defaults.vrtool_config import VrtoolConfig


class TestVrtoolConfig:
    def test_init_vrtool_config_default_values(self):
        _config = VrtoolConfig()
        assert isinstance(_config, VrtoolConfig)

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
        assert _config.plot_reliability_in_time
        assert not _config.plot_measure_reliability
        assert _config.flip_traject
        assert _config.assessment_plot_years == [0, 20, 50]
        assert not _config.geometry_plot
        assert _config.beta_cost_settings == {"symbols": True, "markersize": 10}
        assert isinstance(_config.unit_costs, dict)
        _expected_keys = [
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
        _different_entries = set(_expected_keys) - set(asdict(_config).keys())
        assert not any(
            _different_entries
        ), "Mismatch between expected entries and retrieved: {}".format(
            ",".join(_different_entries)
        )
