from collections import defaultdict

import pandas as pd

from vrtool.common.enums.mechanism_enum import MechanismEnum
from vrtool.flood_defence_system.section_reliability import SectionReliability
from vrtool.orm.io.importers.orm_importer_protocol import OrmImporterProtocol
from vrtool.orm.models.measure_result.measure_result import MeasureResult
from vrtool.orm.models.measure_result.measure_result_mechanism import (
    MeasureResultMechanism,
)
from vrtool.orm.models.measure_result.measure_result_section import MeasureResultSection
from vrtool.orm.models.orm_base_model import OrmBaseModel


class MeasureResultImporter(OrmImporterProtocol):
    @staticmethod
    def import_measure_reliability_df(measure_result: MeasureResult) -> pd.DataFrame:
        """
        Imports all the reliability values of a given `MeasureResult` into a
        `pd.DataFrame`. Said `pd.DataFrame` will have its columns representing
        the available `time` of a `MeasureResult`, the rows as the different
        `Mechanism`, as well as `SectionData` and the values being their resulting
         `beta`.

        Args:
            measure_result (MeasureResult): The measure result whose reliability
            dataframe (`pd.DataFrame`) needs to be imported.

        Returns:
            pd.DataFrame: Dataframe containing reliability relative to the
            measure - section and measure - mechanisms.
        """
        _columns = []
        _section_reliability_dict = defaultdict(list)
        for _smr in measure_result.measure_result_section.order_by(
            MeasureResultSection.time.asc()
        ):
            _columns.append(str(_smr.time))
            _section_reliability_dict["Section"].append(_smr.beta)
            for _mrm in measure_result.measure_result_mechanisms.where(
                MeasureResultMechanism.time == _smr.time
            ):
                _mech_name = MechanismEnum.get_enum(
                    _mrm.mechanism_per_section.mechanism_name
                ).name
                _section_reliability_dict[_mech_name].append(_mrm.beta)

        return pd.DataFrame.from_dict(
            _section_reliability_dict, columns=_columns, orient="index"
        )

    def import_orm(self, measure_result: OrmBaseModel) -> dict:
        _cost = float("nan")
        if any(measure_result.measure_result_section):
            # The measure cost has the same value regardless of the time.
            _cost = measure_result.measure_result_section[0].cost
        _section_reliability = SectionReliability()
        _section_reliability.SectionReliability = self.import_measure_reliability_df(
            measure_result
        )

        # Get measure parameters (dberm, dcrest, target_beta, transition_level, ...).
        _imported_parameters = dict(
            (mrp.name.lower(), mrp.value)
            for mrp in measure_result.measure_result_parameters
        )

        # Set attributes.
        return dict(
            measure_result_id=measure_result.get_id(),
            measure_id=measure_result.measure_per_section.measure.get_id(),
            Cost=_cost,
            Reliability=_section_reliability,
            imported_parameters=_imported_parameters,
            combinable=measure_result.measure_per_section.measure.combinable_type.name,
            reinforcement_type=measure_result.measure_per_section.measure.measure_type.name,
        )
