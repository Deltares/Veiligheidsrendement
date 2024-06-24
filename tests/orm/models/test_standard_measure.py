from tests.orm import with_empty_db_context
from vrtool.orm.models.combinable_type import CombinableType
from vrtool.orm.models.measure import Measure
from vrtool.orm.models.measure_type import MeasureType
from vrtool.orm.models.orm_base_model import OrmBaseModel
from vrtool.orm.models.standard_measure import StandardMeasure


class TestStandardMeasure:
    @with_empty_db_context
    def test_initialize_standard_measure(self):
        # 1. Define test data.
        _measure_type = MeasureType.create(name="dummy_measure_type")
        _combinable_type = CombinableType.create(name="dummy_combinable_type")
        _measure = Measure.create(
            measure_type=_measure_type,
            combinable_type=_combinable_type,
            name="dummy_measure",
        )

        # 2. Run test.
        _standard_measure = StandardMeasure(measure=_measure)

        # 3. Verify expectations
        assert isinstance(_standard_measure, OrmBaseModel)
        assert isinstance(_standard_measure, StandardMeasure)
        assert _standard_measure.measure == _measure
        assert _standard_measure.max_inward_reinforcement == 50
        assert _standard_measure.max_outward_reinforcement == 0
        assert _standard_measure.direction == "Inward"
        assert _standard_measure.crest_step == 0.5
        assert _standard_measure.max_crest_increase == 2
        assert _standard_measure.stability_screen == 0
        assert _standard_measure.prob_of_solution_failure == 1 / 1000
        assert _standard_measure.failure_probability_with_solution == 10**-12
        assert _standard_measure.stability_screen_s_f_increase == 0.2
        assert _standard_measure.transition_level_increase_step == 0.25
        assert _standard_measure.max_pf_factor_block == 1000
        assert _standard_measure.n_steps_block == 4
