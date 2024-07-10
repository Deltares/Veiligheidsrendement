DROP TABLE IF EXISTS sqlitestudio_temp_table;

-- VRTOOL-542 Migration of tables to remove `year` key from `Measure` table.

PRAGMA foreign_keys = 0;

CREATE TABLE sqlitestudio_temp_table AS SELECT *
                                          FROM Measure;

DROP TABLE Measure;

CREATE TABLE Measure (
    id                 INTEGER       NOT NULL
                                     PRIMARY KEY,
    measure_type_id    INTEGER       NOT NULL,
    combinable_type_id INTEGER       NOT NULL,
    name               VARCHAR (128) NOT NULL,
    FOREIGN KEY (
        measure_type_id
    )
    REFERENCES MeasureType (id) ON DELETE CASCADE,
    FOREIGN KEY (
        combinable_type_id
    )
    REFERENCES CombinableType (id) ON DELETE CASCADE
);

INSERT INTO Measure (
                        id,
                        measure_type_id,
                        combinable_type_id,
                        name
                    )
                    SELECT id,
                           measure_type_id,
                           combinable_type_id,
                           name
                      FROM sqlitestudio_temp_table;

DROP TABLE sqlitestudio_temp_table;

CREATE INDEX measure_combinable_type_id ON Measure (
    "combinable_type_id"
);

CREATE INDEX measure_measure_type_id ON Measure (
    "measure_type_id"
);

PRAGMA foreign_keys = 1;
