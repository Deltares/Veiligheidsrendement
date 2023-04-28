from typing import Protocol
from vrtool.orm.models.orm_base_model import OrmBaseModel

class OrmImporterProtocol(Protocol):

    def import_orm(self, orm_model: OrmBaseModel):
        pass