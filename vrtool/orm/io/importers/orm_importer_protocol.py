from typing import Any, Protocol, runtime_checkable

from vrtool.orm.models.orm_base_model import OrmBaseModel


@runtime_checkable
class OrmImporterProtocol(Protocol):
    def import_orm(self, orm_model: OrmBaseModel) -> Any:
        """
        Imports an ORM object model into a `vrtool` domain model (which do not share a base class).

        Args:
            orm_model (OrmBaseModel): An object representing a (database) table's row.

        Returns:
            Any: The mapped object representation in the `vrtool`.
        """
        pass
