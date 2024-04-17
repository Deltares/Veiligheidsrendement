import pytest

from vrtool.common.enums.mechanism_enum import MechanismEnum
from vrtool.decision_making.measures.measure_protocol import MeasureProtocol
from vrtool.decision_making.measures.standard_measures.wall_measures.diaphragm_wall_measure import (
    DiaphragmWallMeasure,
)
from vrtool.decision_making.measures.standard_measures.wall_measures.selfretaining_sheetpile_measure import (
    SelfretainingSheetpileMeasure,
)
from vrtool.flood_defence_system.dike_section import DikeSection
from vrtool.flood_defence_system.mechanism_reliability_collection import (
    MechanismReliabilityCollection,
)

valid_dike_section_cases = [
    dict(
        crest_height=9.98,
        section_length=397,
        piping_properties=dict(
            d_cover=[8.333550637, 4.833550637, 2.833550637],
            h_exit=[4.0, 4.0, 4.0],
        ),
    ),
    dict(
        crest_height=9.93,
        section_length=286,
        piping_properties=dict(
            d_cover=[9.347514637, 5.847514637, 3.847514637],
            h_exit=[3.012, 3.012, 3.012],
        ),
    ),
]


def make_case_invalid(dike_section_case: dict) -> dict:
    _new_dict = dike_section_case | dict()
    _new_dict["piping_properties"] = dict(invalid_values=[4.2, 2.4, 42])
    return _new_dict


invalid_dike_section_cases = list(map(make_case_invalid, valid_dike_section_cases))


class TestSelfretainingSheetpileMeasure:

    def test_initialize(self):
        # 1. Run test.
        _measure = SelfretainingSheetpileMeasure()

        # 2. Verify expectations.
        assert isinstance(_measure, SelfretainingSheetpileMeasure)
        assert isinstance(_measure, DiaphragmWallMeasure)
        assert isinstance(_measure, MeasureProtocol)

    @pytest.fixture
    def indirect_dike_section(self, request: pytest.FixtureRequest) -> DikeSection:
        _dike_section_properties = request.param

        # Define dike section
        class CustomDikeSection(DikeSection):
            piping_properties: dict

        _custom_section = CustomDikeSection()
        _custom_section.crest_height = _dike_section_properties["crest_height"]
        _custom_section.Length = _dike_section_properties["section_length"]
        _custom_section.piping_properties = _dike_section_properties[
            "piping_properties"
        ]

        # Create custom MechanismReliabilityCollection
        _custom_reliability_collection = MechanismReliabilityCollection(
            mechanism=MechanismEnum.PIPING,
            computation_type="TEST",
            computation_years=[0],
            t_0=0,
            measure_year=0,
        )

        # We only need to define year 0
        _custom_reliability_collection.Reliability["0"].Input.input = (
            _custom_section.piping_properties
        )
        _custom_section.section_reliability.failure_mechanisms.add_failure_mechanism_reliability_collection(
            _custom_reliability_collection
        )

        return _custom_section

    @pytest.mark.parametrize(
        "indirect_dike_section",
        [
            pytest.param(valid_dike_section_cases[0], id="Section 1"),
            pytest.param(valid_dike_section_cases[1], id="Section 2"),
        ],
        indirect=True,
    )
    def test_get_dcover_with_valid_data(self, indirect_dike_section: DikeSection):
        # 1. Define test data.
        _measure = SelfretainingSheetpileMeasure()

        # 2. Run test.
        _dcover = _measure._get_dcover(indirect_dike_section)

        # 3. Verify expectations.
        assert _dcover == max(indirect_dike_section.piping_properties["d_cover"])

    @pytest.mark.parametrize(
        "indirect_dike_section",
        [
            pytest.param(invalid_dike_section_cases[0], id="Section 1 - invalid"),
            pytest.param(invalid_dike_section_cases[1], id="Section 2 - invalid"),
        ],
        indirect=True,
    )
    def test_get_dcover_with_invalid_data_returns_default_value(
        self, indirect_dike_section: DikeSection
    ):
        # 1. Define test data.
        _measure = SelfretainingSheetpileMeasure()

        # 2. Run test.
        _dcover = _measure._get_dcover(indirect_dike_section)

        # 3. Verify expectations.
        assert _dcover == 1.0

    @pytest.mark.parametrize(
        "indirect_dike_section",
        [
            pytest.param(valid_dike_section_cases[0], id="Section 1"),
            pytest.param(valid_dike_section_cases[1], id="Section 2"),
        ],
        indirect=True,
    )
    def test_get_maaiveld_with_valid_data(self, indirect_dike_section: DikeSection):
        # 1. Define test data.
        _measure = SelfretainingSheetpileMeasure()

        # 2. Run test.
        _maaiveld = _measure._get_maaiveld(indirect_dike_section)

        # 3. Verify expectations.
        assert _maaiveld == min(indirect_dike_section.piping_properties["h_exit"])

    @pytest.mark.parametrize(
        "indirect_dike_section",
        [
            pytest.param(invalid_dike_section_cases[0], id="Section 1 - invalid"),
            pytest.param(invalid_dike_section_cases[1], id="Section 2 - invalid"),
        ],
        indirect=True,
    )
    def test_get_maaiveld_with_invalid_data_returns_default_value(
        self, indirect_dike_section: DikeSection
    ):
        # 1. Define test data.
        _measure = SelfretainingSheetpileMeasure()

        # 2. Run test.
        _maaiveld = _measure._get_maaiveld(indirect_dike_section)

        # 3. Verify expectations.
        assert _maaiveld == indirect_dike_section.crest_height - 3

    @pytest.mark.parametrize(
        "indirect_dike_section, expected_cost",
        [
            pytest.param(
                valid_dike_section_cases[0],
                8734000,
                id="Section 1",
            ),
            pytest.param(
                valid_dike_section_cases[1],
                6292000,
                id="Section 2",
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
        Traject 38-1, two river sections with selfretaining sheetpile [VRTOOL-344]]

        Thus, this test focuses on only validating the measure costs in a "fast" way,
        instead of having to run the whole `run_measures` step.
        """

        # 1. Define test data.
        _measure = SelfretainingSheetpileMeasure()

        # 2. Run test.
        _total_cost = _measure._calculate_measure_costs(indirect_dike_section)

        # 3. Verify expectations.
        assert _total_cost == expected_cost
