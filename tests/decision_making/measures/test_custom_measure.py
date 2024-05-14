from vrtool.common.dike_traject_info import DikeTrajectInfo
from vrtool.decision_making.measures.custom_measure import CustomMeasure
from vrtool.flood_defence_system.dike_section import DikeSection


class TestCustomMeasure:
    def test_initialize(self):
        # 1./2. Define test data/Run test
        _measure = CustomMeasure()

        # 3. Verify the results
        assert isinstance(_measure, CustomMeasure)

    def test_evaluate_does_not_throw(self):
        # 1. Define test data
        _measure = CustomMeasure()

        # 2. Run test
        _measure.evaluate_measure(
            DikeSection(), DikeTrajectInfo(traject_name=None), False
        )
