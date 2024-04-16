from typing import Any

import pandas as pd

from vrtool.orm.io.importers.orm_importer_protocol import OrmImporterProtocol
from vrtool.orm.models.profile_point import ProfilePoint
from vrtool.orm.models.section_data import SectionData


class GeometryImporter(OrmImporterProtocol):
    def import_orm(self, orm_model: SectionData) -> pd.DataFrame:
        if not orm_model:
            raise ValueError(f"No valid value given for {SectionData.__name__}.")

        records = [self._to_dict(point) for point in orm_model.profile_points]
        geometry = pd.DataFrame.from_records(records)
        geometry.set_index("type", inplace=True, drop=True)

        return geometry

    def _to_dict(self, point: ProfilePoint) -> dict[str, Any]:
        return {
            "type": point.profile_point_type.name,
            "x": point.x_coordinate,
            "z": point.y_coordinate,
        }
