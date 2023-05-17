from vrtool.common.dike_traject_info import DikeTrajectInfo
from vrtool.orm.io.importers.converters import to_valid_float
from vrtool.orm.io.importers.orm_importer_protocol import OrmImporterProtocol
from vrtool.orm.models.dike_traject_info import DikeTrajectInfo as OrmDikeTrajectInfo


class DikeTrajectInfoImporter(OrmImporterProtocol):
    def import_orm(self, orm_model: OrmDikeTrajectInfo) -> DikeTrajectInfo:
        if not orm_model:
            raise ValueError(f"No valid value given for {OrmDikeTrajectInfo.__name__}.")

        return DikeTrajectInfo(
            traject_name=orm_model.traject_name,
            omegaPiping=orm_model.omega_piping,
            omegaStabilityInner=orm_model.omega_stability_inner,
            omegaOverflow=orm_model.omega_overflow,
            aPiping=to_valid_float(orm_model.a_piping),
            bPiping=orm_model.b_piping,
            aStabilityInner=orm_model.a_stability_inner,
            bStabilityInner=orm_model.b_stability_inner,
            beta_max=to_valid_float(orm_model.beta_max),
            Pmax=to_valid_float(orm_model.p_max),
            FloodDamage=to_valid_float(orm_model.flood_damage),
            TrajectLength=to_valid_float(orm_model.traject_length),
        )
