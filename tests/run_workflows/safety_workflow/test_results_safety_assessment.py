from vrtool.run_workflows.safety_workflow.results_safety_assessment import (
    ResultsSafetyAssessment,
)
from vrtool.run_workflows.vrtool_run_result_protocol import VrToolRunResultProtocol


class TestResultsSafetyAssessment:
    def test_init(self):
        _results = ResultsSafetyAssessment()

        assert isinstance(_results, ResultsSafetyAssessment)
        assert isinstance(_results, VrToolRunResultProtocol)
