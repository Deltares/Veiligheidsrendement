import math

import pytest

from vrtool.common.dike_traject_info import DikeTrajectInfo
from vrtool.flood_defence_system.cross_sectional_requirements import (
    CrossSectionalRequirements,
)
from vrtool.flood_defence_system.dike_section import DikeSection
from vrtool.flood_defence_system.section_reliability import SectionReliability


class TestDikeSection:

    def test_initialize(self):
        # 1. Instantiate class.
        _dike_section = DikeSection()

        # 2. Verify expectations.
        assert isinstance(_dike_section, DikeSection)
        assert isinstance(_dike_section.mechanism_data, dict)
        assert isinstance(_dike_section.section_reliability, SectionReliability)
        assert _dike_section.name == ""
        assert _dike_section.TrajectInfo is None
        assert _dike_section.InitialGeometry is None
        assert _dike_section.with_measures == True
        assert math.isnan(_dike_section.crest_height)
        assert math.isnan(_dike_section.cover_layer_thickness)
        assert math.isnan(_dike_section.Length)
        assert math.isnan(_dike_section.pleistocene_level)
        assert math.isnan(_dike_section.flood_damage)
        assert _dike_section.sensitive_fraction_piping == pytest.approx(0.0)
        assert _dike_section.sensitive_fraction_stability_inner == pytest.approx(0.0)

    def test_given_no_traject_info_when_get_cross_sectional_properties_returns_values(self):
        # 1. Define test data.
        _length = 42
        _a_piping = 2.3
        _a_stability_inner = 1.5
        _dike_section = DikeSection(
            Length= _length,
            sensitive_fraction_piping=_a_piping,
            sensitive_fraction_stability_inner=_a_stability_inner
        )
        assert _dike_section.TrajectInfo is None

        # 2. Run test.
        _properties = _dike_section.get_cross_sectional_properties()

        # 3. Verify expectations.
        assert isinstance(_properties, CrossSectionalRequirements)
        assert not _properties.cross_sectional_requirement_per_mechanism
        assert _properties.dike_section_length == _length
        assert _properties.dike_traject_b_piping == pytest.approx(0.0)
        assert _properties.dike_traject_b_stability_inner == pytest.approx(0.0)
        assert _properties.dike_section_a_piping == _a_piping
        assert _properties.dike_section_a_stability_inner == _a_stability_inner

    def test_given_traject_info_when_get_cross_sectional_properties_returns_values(self):
        # 1. Define test data.
        _length = 42
        _a_piping = 2.3
        _a_stability_inner = 1.5
        _b_piping = 1.2
        _b_stability_inner = 1.1
        _traject_info = DikeTrajectInfo(
            traject_name="aTraject",
            bPiping=_b_piping,
            bStabilityInner=_b_stability_inner,
        )
        _dike_section = DikeSection(
            TrajectInfo=_traject_info,
            Length= _length,
            sensitive_fraction_piping=_a_piping,
            sensitive_fraction_stability_inner=_a_stability_inner
        )
        assert _dike_section.TrajectInfo == _traject_info

        # 2. Run test.
        _properties = _dike_section.get_cross_sectional_properties()

        # 3. Verify expectations.
        assert isinstance(_properties, CrossSectionalRequirements)
        assert not _properties.cross_sectional_requirement_per_mechanism
        assert _properties.dike_section_length == _length
        assert _properties.dike_traject_b_piping == _b_piping
        assert _properties.dike_traject_b_stability_inner == _b_stability_inner
        assert _properties.dike_section_a_piping == _a_piping
        assert _properties.dike_section_a_stability_inner == _a_stability_inner
