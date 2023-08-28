from typing import Any, Protocol, runtime_checkable

from vrtool.orm.models.orm_base_model import OrmBaseModel


@runtime_checkable
class OrmExporterProtocol(Protocol):
    def export_dom(self, dom_model: Any) -> list[OrmBaseModel]:
        """
        Exports a domain object model (DOM) into the required ORM objects from the database.

        Args:
            dom_model (Any): A domain object defined in the `vrtool`.

        Returns:
            list[OrmBaseModel]: List of objects created in the database.
        """
        pass
