from random import shuffle

import pytest
from numpy.testing import assert_array_almost_equal

from vrtool.common.enums.mechanism_enum import MechanismEnum
from vrtool.decision_making.measures.measure_protocol import MeasureProtocol
from vrtool.decision_making.measures.standard_measures.revetment_measure import (
    RevetmentMeasure,
)
from vrtool.failure_mechanisms.revetment.relation_grass_revetment import (
    RelationGrassRevetment,
)
from vrtool.failure_mechanisms.revetment.revetment_data_class import RevetmentDataClass
from vrtool.flood_defence_system.dike_section import DikeSection
from vrtool.flood_defence_system.mechanism_reliability_collection import (
    MechanismReliabilityCollection,
)
from vrtool.probabilistic_tools.probabilistic_functions import pf_to_beta


class TestRevetmentMeasure:
    def test_initialize(self):
        # Test to validate it's a parameterless constructor.
        _revetment_measure = RevetmentMeasure()
        assert isinstance(_revetment_measure, RevetmentMeasure)
        assert isinstance(_revetment_measure, MeasureProtocol)

    def test_get_min_beta_target(self):
        # 1. Define test data.
        _test_dike_section = DikeSection()
        _computation_years = [0, 2, 6, 20]
        shuffle(_computation_years)
        _mech_reliability_collection = MechanismReliabilityCollection(
            mechanism=MechanismEnum.REVETMENT,
            computation_type="nvt",
            computation_years=_computation_years,
            t_0=0,
            measure_year=2025,
        )
        for _idx, _computation_year in enumerate(_computation_years):
            _mech_reliability_collection.Reliability[str(_computation_year)].Beta = (
                0.24 + (0.24 * _idx)
            )
        _test_dike_section.section_reliability.failure_mechanisms._failure_mechanisms[
            MechanismEnum.REVETMENT
        ] = _mech_reliability_collection

        # 2. Run test
        _min_beta = RevetmentMeasure()._get_min_beta_target(_test_dike_section)

        # 3. Verify expectations.
        assert (
            _min_beta
            == _mech_reliability_collection.Reliability[
                str(min(_computation_years))
            ].Beta
        )

    @pytest.mark.parametrize(
        "min_beta, vector_size",
        [
            pytest.param(0.42, 4),
            pytest.param(2.4, 6),
        ],
    )
    def test_get_beta_target_vector(self, min_beta: float, vector_size: int):
        """
        Test _get_beta_target_vector with beta_block equal to min_beta

        Args:
            min_beta (float): beta from assessment
            vector_size (int): grid dimension
        """
        # 1. Define test data.
        _beta_block = min_beta
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
        _beta_target_vector = _revetment_measure._get_beta_target_vector(
            min_beta, _beta_block, _pmax
        )

        # 3. Verify expectations.
        assert isinstance(_beta_target_vector, list)
        assert len(_beta_target_vector) == vector_size
        assert_array_almost_equal(_beta_target_vector, _expected_beta_target_vector)

    @pytest.mark.parametrize(
        "min_beta, beta_block, vector_size",
        [
            pytest.param(0.42, 3.0, 4),
            pytest.param(2.4, 3.5, 6),
        ],
    )
    def test_get_beta_target_vector_with_beta_block(
        self, min_beta: float, beta_block: float, vector_size: int
    ):
        """
        Test _get_beta_target_vector with beta_block different from min_beta,
        but smaller than _expected_max (see below)

        Args:
            min_beta (float): beta from assessment
            beta_block (float): lowest beta for block revetment
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
        _expected_step = (_expected_max - beta_block) / (vector_size - 2)
        _expected_beta_target_vector = [min_beta]
        for i in range(vector_size - 1):
            _expected_beta_target_vector.append(beta_block + i * _expected_step)

        # 2. Run test.
        _beta_target_vector = _revetment_measure._get_beta_target_vector(
            min_beta, beta_block, _pmax
        )

        # 3. Verify expectations.
        assert isinstance(_beta_target_vector, list)
        assert len(_beta_target_vector) == vector_size
        assert_array_almost_equal(_beta_target_vector, _expected_beta_target_vector)

    @pytest.mark.parametrize(
        "min_beta, beta_block, vector_size",
        [
            pytest.param(0.42, 8.0, 4),
            pytest.param(2.4, 9.0, 6),
        ],
    )
    def test_get_beta_target_vector_with_large_beta_block(
        self, min_beta: float, beta_block: float, vector_size: int
    ):
        """
        Test _get_beta_target_vector with beta_block different from min_beta,
        and greater than _expected_max (see below)

        Args:
            min_beta (float): beta from assessment
            beta_block (float): lowest beta for block revetment
            vector_size (int): grid dimension
        """
        # 1. Define test data.
        _pmax = 0.0001
        _max_pf_factor_block = 1000
        _revetment_measure = RevetmentMeasure()
        _revetment_measure.parameters["max_pf_factor_block"] = _max_pf_factor_block
        _revetment_measure.parameters["n_steps_block"] = vector_size
        _revetment_measure.parameters["transition_level_increase_step"] = 0.25
        _expected_beta_target_vector = [min_beta]

        # 2. Run test.
        _beta_target_vector = _revetment_measure._get_beta_target_vector(
            min_beta, beta_block, _pmax
        )

        # 3. Verify expectations.
        assert isinstance(_beta_target_vector, list)
        assert len(_beta_target_vector) == 1
        assert_array_almost_equal(_beta_target_vector, _expected_beta_target_vector)

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
