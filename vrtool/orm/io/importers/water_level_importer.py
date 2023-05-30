import openturns as ot
import numpy as np

from vrtool.orm.io.importers.orm_importer_protocol import OrmImporterProtocol
from vrtool.orm.models.section_data import SectionData
from vrtool.orm.models.water_level_data import WaterlevelData
from vrtool.common.hydraulic_loads.load_input import LoadInput
from vrtool.probabilistic_tools.probabilistic_functions import TableDist, beta_to_pf


class WaterLevelImporter(OrmImporterProtocol):
    def import_orm(self, orm_model: SectionData, gridpoints=1000) -> LoadInput:

        years = orm_model.water_level_data_list.select(WaterlevelData.year).distinct()

        load = LoadInput([])
        load.distribution = {}
        for yr in years:
            year = yr.year
            waterLevels = (
                orm_model.water_level_data_list.select()
                .where(WaterlevelData.year == year)
                .order_by(WaterlevelData.water_level.asc())
            )
            waterLevelsCnt = len(waterLevels)
            wls = np.zeros(waterLevelsCnt)
            p_nexc = np.zeros(waterLevelsCnt)
            index = 0
            for waterLevel in waterLevels:
                wls[index] = waterLevel.water_level
                p_nexc[index] = 1.0 - beta_to_pf(waterLevel.beta)
                index += 1

            load.distribution[year] = ot.Distribution(
                TableDist(
                    wls,
                    p_nexc,
                    extrap=True,
                    isload=True,
                    gridpoints=gridpoints,
                )
            )
        return load
