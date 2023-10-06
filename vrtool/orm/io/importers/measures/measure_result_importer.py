from typing import Any
from vrtool.orm.io.importers.orm_importer_protocol import OrmImporterProtocol
from vrtool.orm.models.measure_result.measure_result import MeasureResult
from vrtool.orm.models.measure_result.measure_result_mechanism import (
    MeasureResultMechanism,
)
from vrtool.orm.models.measure_result.measure_result_section import MeasureResultSection
from collections import defaultdict
import pandas as pd

from vrtool.orm.models.orm_base_model import OrmBaseModel


class MeasureResultImporter(OrmImporterProtocol):
    def import_orm(self, measure_result: OrmBaseModel) -> dict:
        _imported_parameters = dict(
            (mrp.name.lower(), mrp.value)
            for mrp in measure_result.measure_result_parameters
        )
        _cost = float("nan")
        _columns = []
        _section_reliability_dict = defaultdict(list)
        for _smr in measure_result.sections_measure_result.order_by(
            MeasureResultSection.time.asc()
        ):
            _columns.append(_smr.time)
            _section_reliability_dict["Section"].append(_smr.beta)
            _cost = _smr.cost
            for _mrm in measure_result.measure_result_mechanisms.where(
                MeasureResultMechanism.time == _smr.time
            ):
                _section_reliability_dict[
                    _mrm.mechanism_per_section.mechanism.name
                ].append(_mrm.beta)

        # Set attributes.
        _section_reliability_df = pd.DataFrame.from_dict(
            _section_reliability_dict, columns=_columns, orient="index"
        )

        return dict(
            measure_result_id=measure_result.get_id(),
            measure_id=measure_result.measure_per_section.measure.get_id(),
            Cost=_cost,
            Reliability=_section_reliability_df,
            imported_parameters=_imported_parameters,
            combinable=measure_result.measure_per_section.measure.combinable_type.name,
            reinforcement_type=measure_result.measure_per_section.measure.measure_type.name,
        )
