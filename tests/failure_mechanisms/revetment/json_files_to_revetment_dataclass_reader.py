from vrtool.failure_mechanisms.revetment.revetment_data_class import RevetmentDataClass
from vrtool.failure_mechanisms.revetment.slope_part import (
    SlopePartBuilder,
    SlopePartProtocol,
    StoneSlopePart,
)
from tests import test_data
from vrtool.failure_mechanisms.revetment.relation_grass_revetment import (
    RelationGrassRevetment,
)
from vrtool.failure_mechanisms.revetment.relation_stone_revetment import (
    RelationStoneRevetment,
)
import json


class JsonFilesToRevetmentDataClassReader:
    def _read_JSON(self, file_name):
        with open(file_name, "r") as openfile:
            json_object = json.load(openfile)
        return json_object

    def _search_slope_part(
        self, slope_parts: list[SlopePartProtocol], slope_part: SlopePartProtocol
    ) -> tuple[bool, SlopePartProtocol]:
        for part in slope_parts:
            if part.begin_part == slope_part.begin_part:
                return [True, part]
        return [False, slope_part]

    def _convert_json_objects(
        self,
        stone_revetment_data: dict,
        grass_revetment_data: dict,
        revetment: RevetmentDataClass,
    ) -> RevetmentDataClass:
        n_sections = stone_revetment_data["aantal deelvakken"]
        for _n_section in range(n_sections):
            try:
                _slope_part = SlopePartBuilder.build(
                    top_layer_type=stone_revetment_data["toplaagtype"][_n_section],
                    begin_part=stone_revetment_data["Zo"][_n_section],
                    end_part=stone_revetment_data["Zb"][_n_section],
                    tan_alpha=stone_revetment_data["tana"][_n_section],
                    top_layer_thickness=stone_revetment_data["D huidig"][_n_section],
                )
            except ValueError as exc_err:
                continue

            [exists, slope] = self._search_slope_part(
                revetment.slope_parts, _slope_part
            )
            if not exists:
                revetment.slope_parts.append(_slope_part)

            if isinstance(slope, StoneSlopePart):
                key = f"deelvak {_n_section}"
                nBeta = len(stone_revetment_data[key]["betaFalen"])
                for m in range(nBeta):
                    rel = RelationStoneRevetment(
                        stone_revetment_data["zichtjaar"],
                        stone_revetment_data[key]["D_opt"][m],
                        stone_revetment_data[key]["betaFalen"][m],
                    )
                    slope.slope_part_relations.append(rel)

        nGrass = len(grass_revetment_data["grasbekleding_begin"])
        for _n_section in range(nGrass):
            rel = RelationGrassRevetment(
                grass_revetment_data["zichtjaar"],
                grass_revetment_data["grasbekleding_begin"][_n_section],
                grass_revetment_data["betaFalen"][_n_section],
            )
            revetment.grass_relations.append(rel)

        return revetment

    def get_revetment_input(self, years: list[int], section: int) -> RevetmentDataClass:
        revetment = RevetmentDataClass()
        for year in years:
            gebuFile = f"revetment/GEBU_{section}_{year}.json"
            dataGEBU = self._read_JSON(test_data / gebuFile)
            zstFile = f"revetment/ZST_{section}_{year}.json"
            dataZST = self._read_JSON(test_data / zstFile)
            revetment = self._convert_json_objects(dataZST, dataGEBU, revetment)
        return revetment
