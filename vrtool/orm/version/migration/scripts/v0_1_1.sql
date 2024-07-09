-- VRTOOL-539:
-- Add table Version
CREATE TABLE IF NOT EXISTS Version (
    id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
    orm_version VARCHAR(128) NOT NULL
);
-- Add initial version if table is empty
INSERT INTO Version (orm_version)
SELECT '0.1.0'
WHERE NOT EXISTS (SELECT * FROM Version); 