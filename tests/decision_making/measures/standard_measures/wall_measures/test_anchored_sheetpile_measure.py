import pandas as pd
import pytest

from vrtool.decision_making.measures.measure_protocol import MeasureProtocol
from vrtool.decision_making.measures.standard_measures.wall_measures.anchored_sheetpile_measure import (
    AnchoredSheetpileMeasure,
)
from vrtool.decision_making.measures.standard_measures.wall_measures.diaphragm_wall_measure import (
    DiaphragmWallMeasure,
)
from vrtool.flood_defence_system.dike_section import DikeSection

valid_dike_section_cases = [
    dict(
        crest_height=9.98,
        section_length=397,
        Initialgeometry=pd.DataFrame.from_dict(
            {"BIT": 3.444}, columns=["z"], orient="index"
        ),
        cover_layer_thickness=7.0,
    ),
    dict(
        crest_height=6.93,
        section_length=286,
        Initialgeometry=pd.DataFrame.from_dict(
            {"BIT": 3.913}, columns=["z"], orient="index"
        ),
        cover_layer_thickness=3.0,
    ),
    dict(
        crest_height=7.90,
        section_length=397,
        Initialgeometry=pd.DataFrame.from_dict(
            {"BIT": 6.0}, columns=["z"], orient="index"
        ),
        cover_layer_thickness=1.0,
    ),
]


class TestAnchoredSheetpileMeasure:
    def test_initialize(self):
        # 1. Run test.
        _measure = AnchoredSheetpileMeasure()

        # 2. Verify expectations.
        assert isinstance(_measure, AnchoredSheetpileMeasure)
        assert isinstance(_measure, DiaphragmWallMeasure)
        assert isinstance(_measure, MeasureProtocol)

    @pytest.fixture
    def indirect_dike_section(self, request: pytest.FixtureRequest) -> DikeSection:  # type: ignore
        _dike_section_properties = request.param

        # Define dike section
        class CustomDikeSection(DikeSection):
            piping_properties: dict

        _custom_section = CustomDikeSection()
        _custom_section.crest_height = _dike_section_properties["crest_height"]
        _custom_section.cover_layer_thickness = _dike_section_properties[
            "cover_layer_thickness"
        ]
        _custom_section.InitialGeometry = _dike_section_properties["Initialgeometry"]
        _custom_section.Length = _dike_section_properties["section_length"]

        return _custom_section

    @pytest.mark.parametrize(
        "indirect_dike_section, expected_cost",
        [
            pytest.param(
                valid_dike_section_cases[0], 8734000, id="Section 1 (length > 20)"
            ),
            pytest.param(
                valid_dike_section_cases[1], 5678844.6, id="Section 2 (length < 20)"
            ),
            pytest.param(
                valid_dike_section_cases[2], 4367000, id="Section 3 (length < 10)"
            ),
        ],
        indirect=["indirect_dike_section"],
    )
    def test_calculate_measure_costs(
        self,
        indirect_dike_section: DikeSection,
        expected_cost: float,
    ):
        """
        The parameters used for this test relate to the exact same content and results
        of the database being used in the acceptance test case:
        Traject 38-1, two river sections with anchored sheetpile [VRTOOL-344]]

        Thus, this test focuses on only validating the measure costs in a "fast" way,
        instead of having to run the whole `run_measures` step.
        """

        # 1. Define test data.
        _measure = AnchoredSheetpileMeasure()

        # 2. Run test.
        _total_cost = _measure._calculate_measure_costs(indirect_dike_section)

        # 3. Verify expectations.
        assert _total_cost == expected_cost
