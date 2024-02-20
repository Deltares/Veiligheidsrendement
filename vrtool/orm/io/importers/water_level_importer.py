import logging

import numpy as np
import openturns as ot

from vrtool.common.hydraulic_loads.load_input import LoadInput
from vrtool.orm.io.importers.orm_importer_protocol import OrmImporterProtocol
from vrtool.orm.models.section_data import SectionData
from vrtool.orm.models.water_level_data import WaterlevelData
from vrtool.probabilistic_tools.probabilistic_functions import TableDist, beta_to_pf


class WaterLevelImporter(OrmImporterProtocol):
    gridpoint: int

    def __init__(self, gridpoints: int) -> None:
        self.gridpoint = gridpoints

    def import_orm(self, orm_model: SectionData) -> LoadInput:

        _available_years = orm_model.water_level_data_list.select(
            WaterlevelData.year
        ).distinct()
        if not any(_available_years):
            logging.warning(
                f"Geen waterstandsdata voor dijkvak {orm_model.section_name}."
            )
            return None

        _load_input = LoadInput([])
        _load_input.distribution = {}
        for yr in _available_years:
            year = yr.year
            _water_level_list = (
                orm_model.water_level_data_list.select()
                .where(WaterlevelData.year == year)
                .order_by(WaterlevelData.water_level.asc())
            )
            _wl_count = len(_water_level_list)
            wls = np.zeros(_wl_count)
            p_nexc = np.zeros(_wl_count)
            index = 0
            for waterLevel in _water_level_list:
                wls[index] = waterLevel.water_level
                p_nexc[index] = 1.0 - beta_to_pf(waterLevel.beta)
                index += 1

            _load_input.distribution[year] = ot.Distribution(
                TableDist(
                    wls,
                    p_nexc,
                    extrap=True,
                    isload=True,
                    gridpoints=self.gridpoint,
                )
            )
        return _load_input
