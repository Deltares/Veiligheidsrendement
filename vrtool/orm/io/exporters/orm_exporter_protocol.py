from typing import Any, Protocol, runtime_checkable

from vrtool.orm.models.orm_base_model import OrmBaseModel


@runtime_checkable
class OrmExporterProtocol(Protocol):
    def export_dom(self, dom_model: Any) -> OrmBaseModel:
        """
        Exports a domain object model (DOM) into an ORM object from the database..

        Args:
            dom_model (Any): A domain object defined in the `vrtool`.

        Returns:
            orm_model (OrmBaseModel): An object representing a (database) table's row.
        """
        pass
