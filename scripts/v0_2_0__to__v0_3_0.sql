-- MIGRATION FILE
-- Contains the required statements to migrate a db created with the scheme in VRTOOL-CORE `v0.2.0`
-- to the scheme defined during developments of VRTOOL-CORE `v0.3.0`
-- Separate statements always with the semicolon `;`

-- Change required during VRTOOL-501
DROP INDEX IF EXISTS "custommeasure_measure_id";
CREATE INDEX "custommeasure_measure_id" ON "CustomMeasure" (
	"measure_id"
);
-- Recreate measure result child tables with FK restriction with on_delete CASCADE (VRTOOL_498)
DROP TABLE IF EXISTS "MeasureResultParameter";
CREATE TABLE "MeasureResultParameter" ("id" INTEGER NOT NULL PRIMARY KEY, "name" VARCHAR(128) NOT NULL, "value" REAL NOT NULL, "measure_result_id" INTEGER NOT NULL, FOREIGN KEY ("measure_result_id") REFERENCES "MeasureResult" ("id") ON DELETE CASCADE);
DROP TABLE IF EXISTS "MeasureResultSection";
CREATE TABLE "MeasureResultSection" ("id" INTEGER NOT NULL PRIMARY KEY, "measure_result_id" INTEGER NOT NULL, "beta" REAL NOT NULL, "time" INTEGER NOT NULL, "cost" REAL NOT NULL, FOREIGN KEY ("measure_result_id") REFERENCES "MeasureResult" ("id") ON DELETE CASCADE);
DROP TABLE IF EXISTS "MeasureResultMechanism";
CREATE TABLE "MeasureResultMechanism" ("id" INTEGER NOT NULL PRIMARY KEY, "measure_result_id" INTEGER NOT NULL, "mechanism_per_section_id" INTEGER NOT NULL, "beta" REAL NOT NULL, "time" INTEGER NOT NULL, FOREIGN KEY ("measure_result_id") REFERENCES "MeasureResult" ("id") ON DELETE CASCADE, FOREIGN KEY ("mechanism_per_section_id") REFERENCES "MechanismPerSection" ("id"));
-- Recreate optimization selected measure table with FK restriction with on_delete CASCADE (VRTOOL_499)
DROP TABLE IF EXISTS "OptimizationSelectedMeasure";
CREATE TABLE "OptimizationSelectedMeasure" ("id" INTEGER NOT NULL PRIMARY KEY, "optimization_run_id" INTEGER NOT NULL, "measure_result_id" INTEGER NOT NULL, "investment_year" INTEGER NOT NULL, FOREIGN KEY ("optimization_run_id") REFERENCES "OptimizationRun" ("id") ON DELETE CASCADE, FOREIGN KEY ("measure_result_id") REFERENCES "MeasureResult" ("id") ON DELETE CASCADE);
-- Recreate custom measure parameter table with FK restriction with on_delete CASCADE (VRTOOL_xxx)
DROP TABLE IF EXISTS "CustomMeasureParameter";
CREATE TABLE "CustomMeasureParameter" ("id" INTEGER NOT NULL PRIMARY KEY, "custom_measure_id" INTEGER NOT NULL, "parameter" VARCHAR(128) NOT NULL, "value" REAL NOT NULL, FOREIGN KEY ("custom_measure_id") REFERENCES "CustomMeasure" ("id") ON DELETE CASCADE);

