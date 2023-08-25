import pytest

from tests.failure_mechanisms.revetment.json_files_to_revetment_dataclass_reader import (
    JsonFilesToRevetmentDataClassReader,
)
from vrtool.failure_mechanisms.revetment.revetment_calculator import RevetmentCalculator


class TestRevetmentCalculatorAssessment:
    @pytest.mark.parametrize(
        "assessment_year, given_years, section_id, ref_values",
        [
            pytest.param(
                2025,
                [2025],
                0,
                [3.6112402089287357, 0.00015236812335053454],
                id="2025_0",
            ),
            pytest.param(
                2100,
                [2100],
                0,
                [3.617047156851664, 0.00014899151576941146],
                id="2100_0",
            ),
            pytest.param(
                2050,
                [2025, 2100],
                0,
                [3.6131758582363784, 0.0001512347051563111],
                id="2050_0",
            ),
        ],
    )
    def test_revetment_calculation(
        self,
        assessment_year: int,
        given_years: list[int],
        section_id: int,
        ref_values: list[float],
    ):
        revetment = JsonFilesToRevetmentDataClassReader().get_revetment_input(
            given_years, section_id
        )

        calc = RevetmentCalculator(revetment, 0)
        [beta, pf] = calc.calculate(assessment_year)

        assert beta == pytest.approx(ref_values[0], rel=1e-8)
        assert pf == pytest.approx(ref_values[1], 1e-8)
