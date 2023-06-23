import json
import pytest

# from vrtool.failure_mechanisms.revetment.revetment_data_class import RevetmentDataClass
from vrtool.failure_mechanisms.revetment.revetment_calculation_assessment import (
    revetmentCalculation,
)
from tests import test_data


def read_JSON(file_name):
    with open(file_name, "r") as openfile:
        json_object = json.load(openfile)
    return json_object


class TestRevetmentAssessmentCalculator:
    def getRevetmentInput(self, year: int, section: int):  # -> RevetmentDataClass:
        gebuFile = f"revetment/GEBU_{section}_{year}.json"
        dataGEBU = read_JSON(test_data / gebuFile)
        zstFile = f"revetment/ZST_{section}_{year}.json"
        dataZST = read_JSON(test_data / zstFile)
        calc = revetmentCalculation()
        betaZST_ini, betaGEBU_ini = calc.evaluate_bekleding(dataZST, dataGEBU)
        beta_ini = calc.beta_comb(betaZST_ini, betaGEBU_ini)
        return beta_ini

    def test_revetment_calculation(self):
        beta_ini = self.getRevetmentInput(2025, 0)

        assert beta_ini == pytest.approx(3.6112402089287357, rel=1e-8)
