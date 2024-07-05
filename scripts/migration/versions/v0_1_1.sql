-- VRTOOL-539:
-- Add table Version
CREATE TABLE IF NOT EXISTS Version (
    id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
    orm_version VARCHAR(128) NOT NULL
);
-- Add initial version
INSERT INTO Version (orm_version) VALUES ('0.1.0');