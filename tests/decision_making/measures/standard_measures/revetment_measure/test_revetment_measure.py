import pytest
from vrtool.common.dike_traject_info import DikeTrajectInfo
from vrtool.decision_making.measures.measure_protocol import MeasureProtocol
from vrtool.decision_making.measures.standard_measures.revetment_measure import (
    RevetmentMeasure,
)
from vrtool.decision_making.measures.standard_measures.revetment_measure.revetment_measure_result_collection import (
    RevetmentMeasureResultCollection,
)
from vrtool.failure_mechanisms.revetment.revetment_data_class import RevetmentDataClass
from vrtool.flood_defence_system.dike_section import DikeSection
from vrtool.flood_defence_system.mechanism_reliability_collection import (
    MechanismReliabilityCollection,
)
from numpy.testing import assert_array_almost_equal


class TestRevetmentMeasure:
    def test_initialize(self):
        # Test to validate it's a parameterless constructor.
        _revetment_measure = RevetmentMeasure()
        assert isinstance(_revetment_measure, RevetmentMeasure)
        assert isinstance(_revetment_measure, MeasureProtocol)

    def test_get_beta_target_vector(self):
        # 1. Define test data.
        _vector_size = 4
        _min_beta = 0.42
        _revetment_measure = RevetmentMeasure()
        _revetment_measure.parameters["max_pf_factor_block"] = 1000
        _revetment_measure.parameters["n_steps_block"] = _vector_size
        _revetment_measure.parameters["transition_level_increase_step"] = 0.25
        _expected_beta_target_vector = [
            0.42,
            0.9738885595475857,
            1.5277771190951714,
            2.081665678642757,
        ]
        # 2. Run test.
        _beta_target_vector = _revetment_measure._get_beta_target_vector(_min_beta, 4.2)

        # 3. Verify expectations.
        assert isinstance(_beta_target_vector, list)
        assert len(_beta_target_vector) == _vector_size
        assert _beta_target_vector[0] == _min_beta
        assert_array_almost_equal(_beta_target_vector, _expected_beta_target_vector)

    def test_get_transition_level_vector(self):
        # 1. Define test data.
        _current_transition_level = 0
        _crest_height = 1
        _transition_level_step = 0.25
        _revetment_measure = RevetmentMeasure()
        _revetment_measure.parameters[
            "transition_level_increase_step"
        ] = _transition_level_step
        _expected_transition_level_vector = [
            0,
            0.25,
            0.5,
            0.75,
        ]
        # 2. Run test.
        _transition_level_vector = _revetment_measure._get_transition_level_vector(
            _current_transition_level, 1
        )

        # 3. Verify expectations.
        assert isinstance(_transition_level_vector, list)
        assert len(_transition_level_vector) == _crest_height / _transition_level_step
        assert _transition_level_vector[0] == _current_transition_level
        assert_array_almost_equal(
            _transition_level_vector, _expected_transition_level_vector
        )

    @pytest.mark.skip(reason="Work in progress.")
    def test_evaluate_measure(self):
        # 1. Define test data.
        _revetment_measure = RevetmentMeasure()
        _revetment_measure.parameters["max_pf_factor_block"] = 1000
        _revetment_measure.parameters["n_steps_block"] = 4
        _revetment_measure.parameters["transition_level_increase_step"] = 0.25
        _dike_section = DikeSection()
        _mech_rel_coll = MechanismReliabilityCollection(
            mechanism="Revetment",
            computation_type="DummyComputationType",
            computation_years=[0, 2, 4],
            t_0=0,
            measure_year=0,
        )
        _dike_section.section_reliability.failure_mechanisms.add_failure_mechanism_reliability_collection(
            _mech_rel_coll
        )
        _mech_rel_coll.Reliability["0"].Input.input[
            "revetment_input"
        ] = RevetmentDataClass()
        _mech_rel_coll.Reliability["0"].Beta = 0
        _mech_rel_coll.Reliability["2"].Beta = 2
        _mech_rel_coll.Reliability["4"].Beta = 4
        _traject = DikeTrajectInfo(traject_name="DummyTraject")
        _traject.Pmax = 3.33e-05
        assert not _revetment_measure.measures

        # 2. Run test.
        _revetment_measure.evaluate_measure(_dike_section, _traject, None)

        # 3. Verify expectations.
        assert isinstance(_revetment_measure.measures, RevetmentMeasureResultCollection)
