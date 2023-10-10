from peewee import FloatField, ForeignKeyField, IntegerField

from vrtool.orm.models.optimization.optimization_step import OptimizationStep
from vrtool.orm.models.orm_base_model import OrmBaseModel, _get_table_name


class OptimizationStepResultSection(OrmBaseModel):
    optimization_step = ForeignKeyField(
        OptimizationStep,
        backref="optimization_step_results_section",
        on_delete="CASCADE",
    )
    beta = FloatField()
    time = IntegerField()
    lcc = FloatField()

    class Meta:
        table_name = _get_table_name(__qualname__)

    @property
    def related_section(self):
        """
        Returns the related orm model of a `SectionData`.

        Returns:
            SectionData: The `SectionData` (`DikeSection`) of this result.
        """
        return (
            self.optimization_step.optimization_selected_measure.measure_result.measure_per_section.section
        )
