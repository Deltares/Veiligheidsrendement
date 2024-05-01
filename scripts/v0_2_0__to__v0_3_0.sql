-- MIGRATION FILE
-- Contains the required statements to migrate a db created with the scheme in VRTOOL-CORE `v0.2.0`
-- to the scheme defined during developments of VRTOOL-CORE `v0.3.0`
-- Separate statements always with the semicolon `;`

DROP INDEX "custommeasure_measure_id";
CREATE INDEX "custommeasure_measure_id" ON "CustomMeasure" (
	"measure_id"
);