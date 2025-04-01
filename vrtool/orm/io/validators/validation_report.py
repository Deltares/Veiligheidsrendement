from dataclasses import dataclass, field


@dataclass
class ValidationError:
    error_message: str
    context_object: object

@dataclass
class ValidationReport:
    context: object
    errors: list[ValidationError] = field(default_factory=list)