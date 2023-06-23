import json
import pytest
import numpy as np

from vrtool.failure_mechanisms.revetment.slope_part import SlopePart
from vrtool.failure_mechanisms.revetment.relation_grass_revetment import (
    RelationGrassRevetment,
)
from vrtool.failure_mechanisms.revetment.relation_stone_revetment import (
    RelationStoneRevetment,
)
from vrtool.failure_mechanisms.revetment.revetment_data_class import RevetmentDataClass
from vrtool.failure_mechanisms.revetment.revetment_calculation_assessment import (
    revetmentCalculation,
)
from tests import test_data


def read_JSON(file_name):
    with open(file_name, "r") as openfile:
        json_object = json.load(openfile)
    return json_object


class TestRevetmentAssessmentCalculator:
    def convertJsonObjects(self, dataZST, dataGEBU) -> RevetmentDataClass:
        revetment = RevetmentDataClass()
        nVakken = dataZST["aantal deelvakken"]
        for n in range(nVakken):
            slopepart = SlopePart(
                dataZST["Zo"][n],
                dataZST["Zb"][n],
                dataZST["tana"][n],
                dataZST["toplaagtype"][n],
                dataZST["D huidig"][n],
            )
            revetment.slope_parts.append(slopepart)
            if slopepart.is_asphalt:
                key = f"deelvak {n}"
                nBeta = len(dataZST[key]["betaFalen"])
                for m in range(nBeta):
                    rel = RelationStoneRevetment(
                        n,
                        dataZST["zichtjaar"],
                        dataZST[key]["D_opt"][m],
                        dataZST[key]["betaFalen"][m],
                    )
                    revetment.stone_relations.append(rel)

        nGrass = len(dataGEBU["grasbekleding_begin"])
        for n in range(nGrass):
            rel = RelationGrassRevetment(
                dataGEBU["zichtjaar"],
                dataGEBU["grasbekleding_begin"][n],
                dataGEBU["betaFalen"][n],
            )
            revetment.grass_relations.append(rel)

        return revetment

    def getRevetmentInput(self, year: int, section: int) -> RevetmentDataClass:
        gebuFile = f"revetment/GEBU_{section}_{year}.json"
        dataGEBU = read_JSON(test_data / gebuFile)
        zstFile = f"revetment/ZST_{section}_{year}.json"
        dataZST = read_JSON(test_data / zstFile)
        revetment = self.convertJsonObjects(dataZST, dataGEBU)
        return revetment

    def test_revetment_calculation(self):
        revetment = self.getRevetmentInput(2025, 0)

        calc = revetmentCalculation()
        betaZST_ini, betaGEBU_ini = calc.evaluate_bekleding(revetment)
        betaZST = np.nanmin(betaZST_ini)
        betaGEBU = np.nanmin(betaGEBU_ini)
        beta_ini = calc.beta_comb(betaZST_ini, betaGEBU_ini)

        assert beta_ini == pytest.approx(3.6112402089287357, rel=1e-8)
        assert betaGEBU == pytest.approx(4.90234375, rel=1e-8)
        assert betaZST == pytest.approx(3.61204720537867, rel=1e-8)
