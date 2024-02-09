from random import shuffle

from numpy.testing import assert_array_almost_equal

from vrtool.common.enums.mechanism_enum import MechanismEnum
from vrtool.decision_making.measures.measure_protocol import MeasureProtocol
from vrtool.decision_making.measures.standard_measures.revetment_measure import (
    RevetmentMeasure,
)
from vrtool.flood_defence_system.dike_section import DikeSection
from vrtool.flood_defence_system.mechanism_reliability_collection import (
    MechanismReliabilityCollection,
)
import pytest


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

    def test_get_beta_target_vector(self):
        # 1. Define test data.
        _vector_size = 4
        _min_beta = 0.42
        _revetment_measure = RevetmentMeasure()
        _revetment_measure.parameters["max_pf_factor_block"] = 1000
        _revetment_measure.parameters["n_steps_block"] = _vector_size
        _revetment_measure.parameters["transition_level_increase_step"] = 0.25
        _expected_beta_target_vector = [
            _min_beta,
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

    @pytest.mark.parametrize(
        "revetment_parameters, expected_result",
        [
            pytest.param(
                dict(current=0, crest_height=1, transition_level_increase_step=0.25),
                [0, 0.25, 0.5, 0.75, 1],
                id="0.0 to 1.0, step 0.25",
            ),
            pytest.param(
                dict(
                    current=2.3, crest_height=4.25, transition_level_increase_step=1.0
                ),
                [2.3, 3.3, 4.25],
                id="2.3 to 4.25, step 1.0, [VRTOOL-330]",
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
        _crest_height = revetment_parameters.pop("crest_height")
        _revetment_measure = RevetmentMeasure()
        _revetment_measure.parameters = revetment_parameters

        # 2. Run test.
        _transition_level_vector = _revetment_measure._get_transition_level_vector(
            _current_transition_level, _crest_height
        )

        # 3. Verify expectations.
        assert isinstance(_transition_level_vector, list)
        assert _transition_level_vector[0] == _current_transition_level
        assert _transition_level_vector[-1] == _crest_height
        assert_array_almost_equal(_transition_level_vector, expected_result)
