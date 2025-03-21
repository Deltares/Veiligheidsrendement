-- VRTOOL-548
DROP TABLE IF EXISTS sqlitestudio_temp_table;

PRAGMA foreign_keys = 0;

CREATE TABLE sqlitestudio_temp_table AS SELECT *
                                          FROM OptimizationStep;

DROP TABLE OptimizationStep;

CREATE TABLE OptimizationStep (
    id                            INTEGER       NOT NULL
                                                PRIMARY KEY,
    step_number                   INTEGER       NOT NULL,
    step_type                     VARCHAR (128),
    selected_primary_measure_id   INTEGER       NOT NULL,
    selected_secondary_measure_id INTEGER,
    total_lcc                     REAL,
    total_risk                    REAL,
    FOREIGN KEY (
        selected_primary_measure_id
    )
    REFERENCES OptimizationSelectedMeasure (id) ON DELETE CASCADE,
    FOREIGN KEY (
        selected_secondary_measure_id
    )
    REFERENCES OptimizationSelectedMeasure (id) ON DELETE CASCADE
);

INSERT INTO OptimizationStep (
                                 id,
                                 step_number,
                                 step_type,
                                 selected_primary_measure_id,
                                 total_lcc,
                                 total_risk
                             )
                             SELECT id,
                                    step_number,
                                    step_type,
                                    optimization_selected_measure_id,
                                    total_lcc,
                                    total_risk
                               FROM sqlitestudio_temp_table;

DROP TABLE sqlitestudio_temp_table;

CREATE INDEX optimizationstep_optimization_selected_primary_measure_id ON OptimizationStep (
    selected_primary_measure_id
);
CREATE INDEX optimizationstep_optimization_selected_secondary_measure_id ON OptimizationStep (
    selected_secondary_measure_id
);

PRAGMA foreign_keys = 1;
