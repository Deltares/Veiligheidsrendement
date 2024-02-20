from vrtool.optimization.measures.measure_as_input_protocol import (
    MeasureAsInputProtocol,
)
from vrtool.optimization.measures.section_as_input import SectionAsInput
from vrtool.run_workflows.optimization_workflow.optimization_input_measures import (
    OptimizationInputMeasures,
)


class MockedMeasure(MeasureAsInputProtocol):
    measure_result_id = 42
    measure_type = None
    combine_type = None
    cost = float("nan")
    discount_rate = float("nan")
    year = 0
    mechanism_year_collection = None
    start_cost = float("nan")


class TestOptimizationInputMeasures:
    def test_measure_id_year_list_doesnot_return_duplicates(self):
        # 1. Define test data
        _measures = [MockedMeasure(), MockedMeasure()]
        assert all((_m.year == 0 and _m.measure_result_id == 42) for _m in _measures)
        _optimization_input_measures = OptimizationInputMeasures(
            vr_config=None,
            selected_traject=None,
            section_input_collection=[
                SectionAsInput(
                    section_name="aSection", traject_name="aTraject", measures=_measures
                )
            ],
        )

        # 2. Run test.
        _id_year_tuple_collection = _optimization_input_measures.measure_id_year_list

        # 3. Verify expectations.
        assert len(_id_year_tuple_collection) == 1
        assert _id_year_tuple_collection[0] == (
            _measures[0].measure_result_id,
            _measures[0].year,
        )
