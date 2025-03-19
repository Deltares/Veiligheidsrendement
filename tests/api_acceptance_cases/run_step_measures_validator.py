from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import pandas as pd
import pytest
from peewee import fn

import vrtool.orm.models as orm
from tests.api_acceptance_cases.run_step_validator_protocol import (
    RunStepValidator,
    _get_database_reference_path,
)
from vrtool.common.enums.measure_type_enum import MeasureTypeEnum
from vrtool.defaults.vrtool_config import VrtoolConfig
from vrtool.orm.io.importers.decision_making.measure_result_importer import (
    MeasureResultImporter,
)
from vrtool.orm.orm_controllers import open_database

OptimizationStepResult = (
    orm.OptimizationStepResultMechanism | orm.OptimizationStepResultSection
)


class RunStepMeasuresValidator(RunStepValidator):
    def validate_preconditions(self, valid_vrtool_config: VrtoolConfig):
        _connected_db = open_database(valid_vrtool_config.input_database_path)

        _custom_measure_result_ids = list(
            _mr.get_id()
            for _mr in orm.MeasureResult.select()
            .join_from(orm.MeasureResult, orm.MeasurePerSection)
            .join_from(orm.MeasurePerSection, orm.Measure)
            .join_from(orm.Measure, orm.MeasureType)
            .where(fn.upper(orm.MeasureType.name) == MeasureTypeEnum.CUSTOM.name)
        )

        assert not any(
            orm.MeasureResult.select().where(
                orm.MeasureResult.id.not_in(_custom_measure_result_ids)
            )
        )
        assert not any(
            orm.MeasureResultParameter.select().where(
                orm.MeasureResultParameter.measure_result_id.not_in(
                    _custom_measure_result_ids
                )
            )
        )

        if not _connected_db.is_closed():
            _connected_db.close()

    def validate_results(self, valid_vrtool_config: VrtoolConfig):
        """
        {
            "section_id":
                "measure_id":
                    { "frozenset[measure_result_with_params]": reliability }
        }
        """

        _reference_database_path = _get_database_reference_path(valid_vrtool_config)

        def load_measures_reliabilities(
            vrtool_db: Path,
        ) -> tuple[
            dict[str, dict[tuple, pd.DataFrame]], dict[str, tuple[float, int, float]]
        ]:
            _connected_db = open_database(vrtool_db)
            _m_reliabilities = defaultdict(dict)
            _m_beta_time_cost = defaultdict(list)
            for _measure_result in orm.MeasureResult.select():
                _measure_per_section = _measure_result.measure_per_section
                _reliability_df = MeasureResultImporter.import_measure_reliability_df(
                    _measure_result
                )
                _available_parameters = frozenset(
                    (mrp.name, mrp.value)
                    for mrp in _measure_result.measure_result_parameters
                )
                _dict_key = (
                    _measure_per_section.measure.name,
                    _measure_per_section.section.section_name,
                )
                if _available_parameters in _m_reliabilities[_dict_key].keys():
                    _keys_values = [f"{k}={v}" for k, v in _available_parameters]
                    _as_string = ", ".join(_keys_values)
                    pytest.fail(
                        "Measure reliability contains twice the same parameters {}.".format(
                            _as_string
                        )
                    )
                _m_reliabilities[_dict_key][_available_parameters] = _reliability_df
                _m_beta_time_cost[_dict_key] = list(
                    sorted(
                        (
                            (_mrs.beta, _mrs.time, _mrs.cost)
                            for _mrs in _measure_result.measure_result_section
                        ),
                        key=lambda btc: btc[1],
                    )
                )
            _connected_db.close()
            return _m_reliabilities, _m_beta_time_cost

        _result_assessment, _result_beta_time_costs = load_measures_reliabilities(
            valid_vrtool_config.input_database_path
        )
        _reference_assessment, _reference_beta_time_costs = load_measures_reliabilities(
            _reference_database_path
        )

        assert any(
            _reference_assessment.items()
        ), "No reference assessments were loaded."

        _errors = []

        # Check costs
        if len(_reference_beta_time_costs) != len(_result_beta_time_costs):
            _errors.append(
                "Not the same length of reference costs ({}) and result costs ({})".format(
                    len(_reference_beta_time_costs), len(_result_beta_time_costs)
                )
            )
        for _ref_key, _ref_beta_time_costs in _reference_beta_time_costs.items():
            _res_beta_time_costs = _result_beta_time_costs.get(_ref_key, list())

            # get only the last values from each tuple
            _ref_costs = [x[2] for x in _ref_beta_time_costs]
            _res_costs = [x[2] for x in _res_beta_time_costs]

            if _ref_costs != _res_costs:
                _key_name = "-".join(_ref_key)

                def beta_year_cost_to_str(cost_collection: list[tuple]) -> str:
                    return ", ".join(
                        [
                            "({}, {}, {})".format(_beta, _year, _cost)
                            for (_beta, _year, _cost) in cost_collection
                        ]
                    )

                _ref_costs_str = beta_year_cost_to_str(_ref_beta_time_costs)
                _res_costs_str = beta_year_cost_to_str(_res_beta_time_costs)
                _errors.append(
                    f"Difference on costs for ({_key_name}), reference: ({_ref_costs_str}), results: ({_res_costs_str})"
                )

        # Check reliability
        for _ref_key, _ref_section_measure_dict in _reference_assessment.items():
            # Iterate over each dictionary entry,
            # which represents ALL the measure results (the values)
            # of a given `MeasurePerSection` (the key).
            _res_section_measure_dict = _result_assessment.get(_ref_key, dict())
            if not any(_res_section_measure_dict.items()):
                _errors.append(
                    "Measure {} = Section {}, have no reliability results.".format(
                        _ref_key[0], _ref_key[1]
                    )
                )
                continue
            for (
                _ref_params,
                _ref_measure_result_reliability,
            ) in _ref_section_measure_dict.items():
                # Iterate over each dictionary entry,
                # which represents the measure reliability results (the values as `pd.DataFrame`)
                # for a given set of parameters represented as `dict` (the keys)
                _res_measure_result_reliability = _res_section_measure_dict.get(
                    _ref_params, pd.DataFrame()
                )
                if _res_measure_result_reliability.empty:
                    _parameters = [f"{k}={v}" for k, v in _ref_params]
                    _parameters_as_str = ", ".join(_parameters)
                    _errors.append(
                        "Measure {} = Section {}, Parameters: {}, have no reliability results".format(
                            _ref_key[0], _ref_key[1], _parameters_as_str
                        )
                    )
                    continue
                pd.testing.assert_frame_equal(
                    _ref_measure_result_reliability, _res_measure_result_reliability
                )
        if _errors:
            pytest.fail("\n".join(_errors))
