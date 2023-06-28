import json
import pytest
import numpy as np

from vrtool.failure_mechanisms.revetment.relation_grass_revetment import (
    RelationGrassRevetment,
)
from vrtool.failure_mechanisms.revetment.relation_stone_revetment import (
    RelationStoneRevetment,
)
from vrtool.failure_mechanisms.revetment.revetment_data_class import RevetmentDataClass
from vrtool.failure_mechanisms.revetment.revetment_calculation_assessment import (
    RevetmentCalculation,
)
from tests import test_data
from vrtool.failure_mechanisms.revetment.slope_part.slope_part_builder import (
    SlopePartBuilder,
)
from vrtool.failure_mechanisms.revetment.slope_part.stone_slope_part import (
    StoneSlopePart,
)


class TestRevetmentAssessmentCalculator:
    def _read_JSON(self, file_name):
        with open(file_name, "r") as openfile:
            json_object = json.load(openfile)
        return json_object

    def _convertJsonObjects(self, dataZST, dataGEBU) -> RevetmentDataClass:
        revetment = RevetmentDataClass()
        n_sections = dataZST["aantal deelvakken"]
        for _n_section in range(n_sections):
            _slope_part = SlopePartBuilder.build(
                top_layer_type=dataZST["toplaagtype"][_n_section],
                begin_part=dataZST["Zo"][_n_section],
                end_part=dataZST["Zb"][_n_section],
                tan_alpha=dataZST["tana"][_n_section],
                top_layer_thickness=dataZST["D huidig"][_n_section],
            )
            revetment.slope_parts.append(_slope_part)
            if isinstance(_slope_part, StoneSlopePart):
                key = f"deelvak {_n_section}"
                nBeta = len(dataZST[key]["betaFalen"])
                for m in range(nBeta):
                    rel = RelationStoneRevetment(
                        dataZST["zichtjaar"],
                        dataZST[key]["D_opt"][m],
                        dataZST[key]["betaFalen"][m],
                    )
                    _slope_part.slope_part_relations.append(rel)

        nGrass = len(dataGEBU["grasbekleding_begin"])
        for _n_section in range(nGrass):
            rel = RelationGrassRevetment(
                dataGEBU["zichtjaar"],
                dataGEBU["grasbekleding_begin"][_n_section],
                dataGEBU["betaFalen"][_n_section],
            )
            revetment.grass_relations.append(rel)

        return revetment

    def _get_revetment_input(self, year: int, section: int) -> RevetmentDataClass:
        gebuFile = f"revetment/GEBU_{section}_{year}.json"
        dataGEBU = self._read_JSON(test_data / gebuFile)
        zstFile = f"revetment/ZST_{section}_{year}.json"
        dataZST = self._read_JSON(test_data / zstFile)
        revetment = self._convertJsonObjects(dataZST, dataGEBU)
        return revetment

    @pytest.mark.parametrize(
        "year, section_id, ref_values",
        [
            pytest.param(
                2025, 0, [3.6112402089287357, 4.90234375, 3.61204720537867], id="2025_0"
            )
        ],
    )
    def test_revetment_calculation(
        self, year: int, section_id: int, ref_values: list[float]
    ):
        revetment = self._get_revetment_input(year, section_id)

        calc = RevetmentCalculation(revetment)
        betaZST_ini, betaGEBU_ini = calc.calculate(None)
        betaZST = np.nanmin(betaZST_ini)
        beta_ini = calc.beta_comb(betaZST_ini, betaGEBU_ini)

        assert beta_ini == pytest.approx(ref_values[0], rel=1e-8)
        assert betaGEBU_ini == pytest.approx(ref_values[1], rel=1e-8)
        assert betaZST == pytest.approx(ref_values[2], rel=1e-8)
