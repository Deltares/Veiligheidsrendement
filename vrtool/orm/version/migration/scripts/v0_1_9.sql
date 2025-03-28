-- VRTOOL-601
DROP TABLE IF EXISTS sqlitestudio_temp_table;

PRAGMA foreign_keys = 0;

CREATE TABLE sqlitestudio_temp_table AS SELECT *
                                          FROM OptimizationStep;

DROP TABLE OptimizationStep;

CREATE TABLE OptimizationStep (
    id                               INTEGER       NOT NULL
                                                   PRIMARY KEY,
    optimization_selected_measure_id INTEGER       NOT NULL,
    step_number                      INTEGER       NOT NULL,
    step_type                        VARCHAR (128),
    total_lcc                        REAL,
    total_risk                       REAL,
    FOREIGN KEY (
        optimization_selected_measure_id
    )
    REFERENCES OptimizationSelectedMeasure (id) ON DELETE CASCADE
);

INSERT INTO OptimizationStep (
                                 id,
                                 optimization_selected_measure_id,
                                 step_number,
                                 total_lcc,
                                 total_risk
                             )
                             SELECT id,
                                    optimization_selected_measure_id,
                                    step_number,
                                    total_lcc,
                                    total_risk
                               FROM sqlitestudio_temp_table;

DROP TABLE sqlitestudio_temp_table;

CREATE INDEX optimizationstep_optimization_selected_measure_id ON OptimizationStep (
    "optimization_selected_measure_id"
);

PRAGMA foreign_keys = 1;
