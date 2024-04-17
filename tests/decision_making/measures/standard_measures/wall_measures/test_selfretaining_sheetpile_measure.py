from vrtool.decision_making.measures.measure_protocol import MeasureProtocol
from vrtool.decision_making.measures.standard_measures.wall_measures.diaphragm_wall_measure import (
    DiaphragmWallMeasure,
)
from vrtool.decision_making.measures.standard_measures.wall_measures.selfretaining_sheetpile_measure import (
    SelfretainingSheetpileMeasure,
)
from vrtool.flood_defence_system.dike_section import DikeSection


class TestSelfretainingSheetpileMeasure:

    def test_initialize(self):
        # 1. Run test.
        _measure = SelfretainingSheetpileMeasure()

        # 2. Verify expectations.
        assert isinstance(_measure, SelfretainingSheetpileMeasure)
        assert isinstance(_measure, DiaphragmWallMeasure)
        assert isinstance(_measure, MeasureProtocol)

    def test_calculate_measure_costs(self):
        # 1. Define test data.
        _measure = SelfretainingSheetpileMeasure()
        _dummy_section = DikeSection()

        # 2. Run test.
        _total_cost = _measure._calculate_measure_costs(_dummy_section)

        # 3. Verify expectations.
        assert _total_cost > 0.0
