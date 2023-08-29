from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class OrmExporterProtocol(Protocol):
    def export_dom(self, dom_model: Any) -> None:
        """
        Exports a domain object model (DOM) into the required ORM objects from the database.

        Args:
            dom_model (Any): A domain object defined in the `vrtool`.
        """
        pass
