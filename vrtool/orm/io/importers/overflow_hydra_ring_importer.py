import pandas as pd

from vrtool.common.enums import MechanismEnum
from vrtool.failure_mechanisms.mechanism_input import MechanismInput
from vrtool.orm.io.importers.orm_importer_protocol import OrmImporterProtocol
from vrtool.orm.models.computation_scenario import ComputationScenario
from vrtool.orm.models.mechanism_table import MechanismTable


class OverFlowHydraRingImporter(OrmImporterProtocol):
    def _get_crest_height_beta(
        self, mechanism_table_rows: list[MechanismTable], scenario_name: str
    ) -> pd.DataFrame:
        def get_records_by_year(rows: list[MechanismTable]) -> dict[int, list[tuple]]:
            lookup = {}
            for row in rows:
                if not lookup.get(row.year, None):
                    lookup[row.year] = list()

                lookup[row.year].append((row.value, row.beta))

            return lookup

        def is_crest_height_equal(
            crest_height_beta_data: list[pd.DataFrame],
            crest_height_beta_mapping: list[tuple],
        ) -> bool:
            return all(
                crest_height_beta_data[-1].index.values
                == [
                    crest_height_beta_map[0]
                    for crest_height_beta_map in crest_height_beta_mapping
                ]
            )

        records = get_records_by_year(mechanism_table_rows)
        crest_height_beta_data = []
        for count, year in enumerate(records.keys()):
            crest_height_beta_mapping = [r for r in records[year]]

            if count > 0 and not is_crest_height_equal(
                crest_height_beta_data, crest_height_beta_mapping
            ):
                raise ValueError(
                    f"Crest heights not equal for scenario {scenario_name}."
                )

            crest_height_row_name = "Crest_Height"
            beta_column_name = "Beta"
            beta_for_year = pd.DataFrame(
                {
                    crest_height_row_name: [i[0] for i in crest_height_beta_mapping],
                    beta_column_name: [i[1] for i in crest_height_beta_mapping],
                }
            )
            beta_for_year.set_index(crest_height_row_name, inplace=True, drop=True)
            beta_for_year.rename(columns={beta_column_name: str(year)}, inplace=True)
            crest_height_beta_data.append(beta_for_year)

        return pd.concat(crest_height_beta_data, axis=1)

    def import_orm(self, orm_model: ComputationScenario) -> MechanismInput:
        if not orm_model:
            raise ValueError(
                f"No valid value given for {ComputationScenario.__name__}."
            )

        mechanism_input = MechanismInput(MechanismEnum.OVERFLOW)
        mechanism_input.input["hc_beta"] = self._get_crest_height_beta(
            orm_model.mechanism_tables.select(), orm_model.scenario_name
        )
        mechanism_input.input[
            "h_crest"
        ] = orm_model.mechanism_per_section.section.crest_height
        mechanism_input.input[
            "d_crest"
        ] = orm_model.mechanism_per_section.section.annual_crest_decline

        return mechanism_input
