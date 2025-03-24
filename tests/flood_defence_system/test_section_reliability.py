from typing import Callable, Iterator

import pandas as pd
import pytest

from vrtool.common.enums.computation_type_enum import ComputationTypeEnum
from vrtool.common.enums.mechanism_enum import MechanismEnum
from vrtool.flood_defence_system.mechanism_reliability_collection import (
    MechanismReliabilityCollection,
)
from vrtool.flood_defence_system.section_reliability import SectionReliability


class TestSectionReliability:
    def test_init_sets_properties(self):
        # Call
        _section_reliability = SectionReliability()

        # Assert
        assert len(_section_reliability.failure_mechanisms.get_available_mechanisms()) == 0
        assert isinstance(_section_reliability.SectionReliability, pd.DataFrame)
        assert _section_reliability.SectionReliability.empty is True


    @pytest.fixture(name="section_reliability_builder")
    def _get_section_reliability_builder_fixture(self) -> Iterator[Callable[[MechanismEnum], SectionReliability]]:
        _years = [0, 10]
        def _get_mrc(mechanism: MechanismEnum) -> MechanismReliabilityCollection:
            _mrc = MechanismReliabilityCollection(
                mechanism=mechanism,
                computation_type=ComputationTypeEnum.SIMPLE,
                computation_years=_years,
                t_0=_years[0],
                measure_year=5
            )
            for _yt in _years:
                _mrc.Reliability[str(_yt)].Pf = 0.42 / (100 * max(1, _yt))
            return _mrc
        
        def build_section_reliability_for_mechanism(mechanism: MechanismEnum) -> SectionReliability:
            _section_reliability = SectionReliability()
            _section_reliability.failure_mechanisms.add_failure_mechanism_reliability_collection(
                _get_mrc(mechanism)
            )
            return _section_reliability

        yield build_section_reliability_for_mechanism
        
    @pytest.mark.parametrize("mechanism, expected_values",
                             [
        pytest.param(
            MechanismEnum.PIPING,
            [0.791935, 1.981413],
            id=str(MechanismEnum.PIPING)),
        pytest.param(
            MechanismEnum.STABILITY_INNER,
            [0.561045, 1.835016],
            id=str(MechanismEnum.STABILITY_INNER)),
    ])
    def test_calculate_section_reliability(self, mechanism: MechanismEnum, expected_values: list[float], section_reliability_builder: Callable[[MechanismEnum], SectionReliability]):
        # 1. Define test data.
        _expected_result = pd.DataFrame.from_dict(
            {
                str(mechanism): expected_values,
                "Section": expected_values,
            }, columns=['0', '10'], orient="index"
        )
        _cross_sectional_properties = dict(
                length=42,
                a_section_piping=1.5,
                a_section_stability_inner=2.3,
                b_piping=1.1,
                b_stability_inner=1.2
            )
        _section_reliability = section_reliability_builder(mechanism)
        assert isinstance(_section_reliability, SectionReliability)
        assert isinstance(_section_reliability.SectionReliability, pd.DataFrame)
        assert _section_reliability.SectionReliability.empty is True

        # 2. Run test.
        _section_reliability.calculate_section_reliability(_cross_sectional_properties)

        # 3. Verify expectations.
        assert isinstance(_section_reliability.SectionReliability, pd.DataFrame)
        assert _section_reliability.SectionReliability.empty is False
        pd.testing.assert_frame_equal(_section_reliability.SectionReliability, _expected_result, check_dtype=False)