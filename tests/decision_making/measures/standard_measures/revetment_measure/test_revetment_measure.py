from random import shuffle

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from vrtool.common.enums.mechanism_enum import MechanismEnum
from vrtool.decision_making.measures.measure_protocol import MeasureProtocol
from vrtool.decision_making.measures.standard_measures.revetment_measure import (
    RevetmentMeasure,
)
from vrtool.decision_making.measures.standard_measures.revetment_measure.revetment_measure_result_collection import (
    RevetmentMeasureResultCollection,
)
from vrtool.decision_making.measures.standard_measures.revetment_measure.revetment_measure_section_reliability import (
    RevetmentMeasureSectionReliability,
)
from vrtool.failure_mechanisms.revetment.relation_grass_revetment import (
    RelationGrassRevetment,
)
from vrtool.failure_mechanisms.revetment.revetment_data_class import RevetmentDataClass
from vrtool.flood_defence_system.dike_section import DikeSection
from vrtool.flood_defence_system.mechanism_reliability_collection import (
    MechanismReliabilityCollection,
)
from vrtool.flood_defence_system.section_reliability import SectionReliability
from vrtool.probabilistic_tools.probabilistic_functions import pf_to_beta


class TestRevetmentMeasure:
    def test_initialize(self):
        # Test to validate it's a parameterless constructor.
        _revetment_measure = RevetmentMeasure()
        assert isinstance(_revetment_measure, RevetmentMeasure)
        assert isinstance(_revetment_measure, MeasureProtocol)

    def test_get_beta_max(self):
        """
        test for the method _get_beta_max
        """
        # 1. Define test data.
        pmax = 0.001
        _max_pf_factor_block = 1000
        _revetment_measure = RevetmentMeasure()
        _revetment_measure.parameters["max_pf_factor_block"] = _max_pf_factor_block

        # 2. Run test.
        _beta_max = _revetment_measure._get_beta_max(pmax)

        # 3. Verify expectations.
        _expected = pf_to_beta(1e-6)
        assert _beta_max == pytest.approx(_expected)

    @pytest.mark.parametrize(
        "min_beta, vector_size",
        [
            pytest.param(0.42, 4),
            pytest.param(2.4, 6),
        ],
    )
    def test_get_beta_target_vector(self, min_beta: float, vector_size: int):
        """
        Test _get_beta_target_vector

        Args:
            min_beta (float): beta from assessment
            vector_size (int): grid dimension
        """
        # 1. Define test data.
        _pmax = 0.0001
        _max_pf_factor_block = 1000
        _revetment_measure = RevetmentMeasure()
        _revetment_measure.parameters["max_pf_factor_block"] = _max_pf_factor_block
        _revetment_measure.parameters["n_steps_block"] = vector_size
        _revetment_measure.parameters["transition_level_increase_step"] = 0.25
        _expected_max = pf_to_beta(_pmax / _max_pf_factor_block)
        _expected_step = (_expected_max - min_beta) / (vector_size - 1)
        _expected_beta_target_vector = []
        for i in range(vector_size):
            _expected_beta_target_vector.append(min_beta + i * _expected_step)

        # 2. Run test.
        _max_beta = _revetment_measure._get_beta_max(_pmax)
        _beta_target_vector = _revetment_measure._get_beta_target_vector(
            min_beta, _max_beta
        )

        # 3. Verify expectations.
        assert isinstance(_beta_target_vector, np.ndarray)
        assert len(_beta_target_vector) == vector_size
        assert_array_almost_equal(_beta_target_vector, _expected_beta_target_vector)

    @pytest.mark.parametrize(
        "min_beta, vector_size, computed_vector_size",
        [
            pytest.param(0.42, 100, 17),
            pytest.param(2.4, 100, 10),
            pytest.param(7.0, 4, 1),
            pytest.param(5.11, 4, 1),
            pytest.param(5.05, 4, 2),
        ],
    )
    def test_get_beta_target_vector_with_reduced_step(
        self, min_beta: float, vector_size: int, computed_vector_size: int
    ):
        """
        Test automatic scaling of stepsize _get_beta_target_vector

        Args:
            min_beta (float): beta from assessment
            computed_vector_size (int): reduced grid dimension
        """
        # 1. Define test data.
        _pmax = 0.0001
        _max_pf_factor_block = 1000
        _revetment_measure = RevetmentMeasure()
        _revetment_measure.parameters["max_pf_factor_block"] = _max_pf_factor_block
        _revetment_measure.parameters["n_steps_block"] = vector_size
        _revetment_measure.parameters["transition_level_increase_step"] = 0.25

        # 2. Run test.
        _max_beta = _revetment_measure._get_beta_max(_pmax)
        _beta_target_vector = _revetment_measure._get_beta_target_vector(
            min_beta, _max_beta
        )

        # 3. Verify expectations.
        assert isinstance(_beta_target_vector, np.ndarray)
        assert len(_beta_target_vector) == computed_vector_size
        if computed_vector_size > 2:
            _step_size = _beta_target_vector[1] - _beta_target_vector[0]
            assert _step_size == pytest.approx(_revetment_measure.minimal_stepsize, 0.2)
        elif computed_vector_size == 2:
            _step_size = _beta_target_vector[1] - _beta_target_vector[0]
            assert _beta_target_vector[0] == pytest.approx(min_beta, 1e-12)
            assert _beta_target_vector[1] == pytest.approx(_max_beta, 1e-12)
            assert _step_size >= _revetment_measure.margin_min_max_beta
        else:
            assert _beta_target_vector[0] == pytest.approx(min_beta, 1e-12)

    @pytest.mark.parametrize(
        "revetment_parameters, expected_result",
        [
            pytest.param(
                dict(
                    current=0,
                    max_level=0.9,
                    crest_height=1.0,
                    transition_level_increase_step=0.25,
                ),
                [0, 0.25, 0.5, 0.75, 0.9],
                id="0.0 to 1.0, step 0.25",
            ),
            pytest.param(
                dict(
                    current=2.3,
                    max_level=4.25,
                    crest_height=4.3,
                    transition_level_increase_step=1.0,
                ),
                [2.3, 3.3, 4.25],
                id="2.3 to 4.3, step 1.0, [VRTOOL-330]",
            ),
            pytest.param(
                dict(
                    current=0,
                    max_level=7 * 0.1,
                    crest_height=0.7,
                    transition_level_increase_step=0.1,
                ),
                [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
                id="0.0 to 0.7, step 0.1",
            ),
        ],
    )
    def test_get_transition_level_vector(
        self, revetment_parameters: dict, expected_result: list[float]
    ):
        """
        This test represents the validation of VRTOOL-330
        """

        # 1. Define test data.
        _current_transition_level = revetment_parameters.pop("current")
        _max_transition_level = revetment_parameters.pop("max_level")

        class MockedRevetmentDataClass(RevetmentDataClass):
            @property
            def current_transition_level(self) -> float:
                return _current_transition_level

        def create_relation_grass_revetment(
            transition_level: float,
        ) -> RevetmentDataClass:
            return RelationGrassRevetment(
                year=2021, transition_level=transition_level, beta=4.2
            )

        _revetment_dc = MockedRevetmentDataClass(
            grass_relations=[
                create_relation_grass_revetment(_current_transition_level),
                create_relation_grass_revetment(_max_transition_level),
            ]
        )
        _crest_height = revetment_parameters.pop("crest_height")
        _revetment_measure = RevetmentMeasure()
        _revetment_measure.parameters = revetment_parameters

        # 2. Run test.
        _transition_level_vector = _revetment_measure._get_transition_level_vector(
            _revetment_dc, _crest_height
        )

        # 3. Verify expectations.
        assert isinstance(_transition_level_vector, list)
        assert _transition_level_vector[0] == _current_transition_level
        assert _transition_level_vector[-1] == _max_transition_level
        assert_array_almost_equal(_transition_level_vector, expected_result)

    def _get_revetment_measure_result(
        self, cost: float, beta: float
    ) -> RevetmentMeasureSectionReliability:
        """
        returns a RevetmentMeasureSectionReliability

        Args:
            cost (float): the cost of this measure
            beta (float): the resulting beta of this measure

        Returns:
            RevetmentMeasureSectionReliability: the measure result
        """

        class MockedSectionReliability(SectionReliability):
            """
            test class to get a section reliability
            """

            def __init__(self):
                self.SectionReliability = {"0": {"Section": beta}}

        _msr = RevetmentMeasureSectionReliability()
        _msr.cost = cost
        _msr.section_reliability = MockedSectionReliability()
        return _msr

    def test_filtering_keep_all(self):
        """
        test the filtering of revetment measures with a situation that no measure is filtered
        """
        msr = RevetmentMeasure()
        msr.measures = RevetmentMeasureResultCollection()
        msr.measures.result_collection.append(
            self._get_revetment_measure_result(23.0, 3.0)
        )
        msr.measures.result_collection.append(
            self._get_revetment_measure_result(24.0, 3.1)
        )
        msr.measures.result_collection.append(
            self._get_revetment_measure_result(25.0, 3.2)
        )
        msr.measures.result_collection.append(
            self._get_revetment_measure_result(26.0, 3.3)
        )
        msr._get_filtered_measures()
        assert len(msr.measures.result_collection) == 4

    def test_filtering_removed_two(self):
        """
        test the filtering of revetment measures with a situation where two measures are filtered away
        """
        msr = RevetmentMeasure()
        msr.measures = RevetmentMeasureResultCollection()
        msr.measures.result_collection.append(
            self._get_revetment_measure_result(23.1, 3.0)
        )
        msr.measures.result_collection.append(
            self._get_revetment_measure_result(24.0, 2.9)
        )
        msr.measures.result_collection.append(
            self._get_revetment_measure_result(25.0, 3.2)
        )
        msr.measures.result_collection.append(
            self._get_revetment_measure_result(26.0, 3.3)
        )
        # now add one that lower cost and higher beta than the first two:
        msr.measures.result_collection.append(
            self._get_revetment_measure_result(23.0, 3.1)
        )
        msr._get_filtered_measures()
        assert len(msr.measures.result_collection) == 3

    def test_filtering_two_equal_measures(self):
        """
        test the filtering of revetment measures with two equal measures
        """
        msr = RevetmentMeasure()
        msr.measures = RevetmentMeasureResultCollection()
        msr.measures.result_collection.append(
            self._get_revetment_measure_result(23.1, 3.0)
        )
        msr.measures.result_collection.append(
            self._get_revetment_measure_result(23.1, 3.0)
        )
        msr._get_filtered_measures()
        assert len(msr.measures.result_collection) == 1

    def test_filtering_three_almost_equal_measures(self):
        """
        test the filtering of revetment measures with three almost equal measures
        """
        msr = RevetmentMeasure()
        _tol_beta = msr.tol_abs_beta_in_filtering
        _tol_costs = msr.tol_rel_costs_in_filtering
        msr.measures = RevetmentMeasureResultCollection()
        msr.measures.result_collection.append(
            self._get_revetment_measure_result(23.1, 3.0)
        )
        msr.measures.result_collection.append(
            self._get_revetment_measure_result(23.1, 3.0 + 0.9 * _tol_beta)
        )
        msr.measures.result_collection.append(
            self._get_revetment_measure_result(23.1 * (1 + 0.9 * _tol_costs), 3.0)
        )
        msr._get_filtered_measures()
        assert len(msr.measures.result_collection) == 1
