DROP TABLE IF EXISTS sqlitestudio_temp_table;

-- VRTOOL-546 Renamed `CustomMeasureDetail` property `year` to `time`.
-- CustomMeasureDetail

PRAGMA foreign_keys = 0;

CREATE TABLE sqlitestudio_temp_table AS SELECT *
                                          FROM CustomMeasureDetail;

DROP TABLE CustomMeasureDetail;

CREATE TABLE CustomMeasureDetail (
    id                       INTEGER NOT NULL
                                     PRIMARY KEY,
    measure_id               INTEGER NOT NULL,
    mechanism_per_section_id INTEGER NOT NULL,
    cost                     REAL,
    beta                     REAL,
    time                     INTEGER NOT NULL,
    FOREIGN KEY (
        measure_id
    )
    REFERENCES Measure (id) ON DELETE CASCADE,
    FOREIGN KEY (
        mechanism_per_section_id
    )
    REFERENCES MechanismPerSection (id) ON DELETE CASCADE
);

INSERT INTO CustomMeasureDetail (
                                    id,
                                    measure_id,
                                    mechanism_per_section_id,
                                    cost,
                                    beta,
                                    time
                                )
                                SELECT id,
                                       measure_id,
                                       mechanism_per_section_id,
                                       cost,
                                       beta,
                                       year
                                  FROM sqlitestudio_temp_table;

DROP TABLE sqlitestudio_temp_table;

CREATE INDEX custommeasuredetail_measure_id ON CustomMeasureDetail (
    "measure_id"
);

CREATE INDEX custommeasuredetail_mechanism_id ON CustomMeasureDetail (
    mechanism_per_section_id
);

PRAGMA foreign_keys = 1;
