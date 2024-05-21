-- MIGRATION FILE
-- Contains the required statements to migrate a db created with the scheme in VRTOOL-CORE `v0.2.0`
-- to the scheme defined during developments of VRTOOL-CORE `v0.3.0`
-- Separate statements always with the semicolon `;`
DROP TABLE IF EXISTS sqlitestudio_temp_table;

-- Change required during VRTOOL-501
DROP INDEX "custommeasure_measure_id";
CREATE INDEX "custommeasure_measure_id" ON "CustomMeasure" (
	"measure_id"
);

-- Changes required for VRTOOL-496
-- CustomMeasure
PRAGMA foreign_keys = 0;

CREATE TABLE sqlitestudio_temp_table AS SELECT * FROM CustomMeasure;
DROP TABLE CustomMeasure;

CREATE TABLE CustomMeasure (
    id           INTEGER NOT NULL
                         PRIMARY KEY,
    measure_id   INTEGER NOT NULL,
    mechanism_id INTEGER NOT NULL,
    cost         REAL,
    beta         REAL,
    year         INTEGER NOT NULL,
    FOREIGN KEY (
        measure_id
    )
    REFERENCES Measure (id) ON DELETE CASCADE,
    FOREIGN KEY (
        mechanism_id
    )
    REFERENCES Mechanism (id) ON DELETE CASCADE
);

INSERT INTO CustomMeasure (
                              id,
                              measure_id,
                              mechanism_id,
                              cost,
                              beta,
                              year
                          )
                          SELECT id,
                                 measure_id,
                                 mechanism_id,
                                 cost,
                                 beta,
                                 year
                            FROM sqlitestudio_temp_table;

DROP TABLE sqlitestudio_temp_table;
CREATE INDEX custommeasure_measure_id ON CustomMeasure (
    "measure_id"
);
CREATE INDEX custommeasure_mechanism_id ON CustomMeasure (
    "mechanism_id"
);
PRAGMA foreign_keys = 1;

-- CustomMeasureParameter
PRAGMA foreign_keys = 0;

CREATE TABLE sqlitestudio_temp_table AS SELECT *
                                          FROM CustomMeasureParameter;

DROP TABLE CustomMeasureParameter;

CREATE TABLE CustomMeasureParameter (
    id                INTEGER       NOT NULL
                                    PRIMARY KEY,
    custom_measure_id INTEGER       NOT NULL,
    parameter         VARCHAR (128) NOT NULL,
    value             REAL          NOT NULL,
    FOREIGN KEY (
        custom_measure_id
    )
    REFERENCES CustomMeasure (id) ON DELETE CASCADE
);

INSERT INTO CustomMeasureParameter (
                                       id,
                                       custom_measure_id,
                                       parameter,
                                       value
                                   )
                                   SELECT id,
                                          custom_measure_id,
                                          parameter,
                                          value
                                     FROM sqlitestudio_temp_table;

DROP TABLE sqlitestudio_temp_table;

CREATE INDEX custommeasureparameter_custom_measure_id ON CustomMeasureParameter (
    "custom_measure_id"
);

PRAGMA foreign_keys = 1;

-- Measure
PRAGMA foreign_keys = 0;

CREATE TABLE sqlitestudio_temp_table AS SELECT * FROM Measure;
DROP TABLE Measure;
CREATE TABLE Measure (
    id                 INTEGER       NOT NULL
                                     PRIMARY KEY,
    measure_type_id    INTEGER       NOT NULL,
    combinable_type_id INTEGER       NOT NULL,
    name               VARCHAR (128) NOT NULL,
    year               INTEGER       NOT NULL,
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
                        name,
                        year
                    )
                    SELECT id,
                           measure_type_id,
                           combinable_type_id,
                           name,
                           year
                      FROM sqlitestudio_temp_table;

DROP TABLE sqlitestudio_temp_table;
CREATE INDEX measure_combinable_type_id ON Measure (
    "combinable_type_id"
);
CREATE INDEX measure_measure_type_id ON Measure (
    "measure_type_id"
);
PRAGMA foreign_keys = 1;

-- MeasurePerSection
PRAGMA foreign_keys = 0;

CREATE TABLE sqlitestudio_temp_table AS SELECT *
                                          FROM MeasurePerSection;

DROP TABLE MeasurePerSection;

CREATE TABLE MeasurePerSection (
    id         INTEGER NOT NULL
                       PRIMARY KEY,
    section_id INTEGER NOT NULL,
    measure_id INTEGER NOT NULL,
    FOREIGN KEY (
        section_id
    )
    REFERENCES SectionData (id) ON DELETE CASCADE,
    FOREIGN KEY (
        measure_id
    )
    REFERENCES Measure (id) ON DELETE CASCADE
);

INSERT INTO MeasurePerSection (
                                  id,
                                  section_id,
                                  measure_id
                              )
                              SELECT id,
                                     section_id,
                                     measure_id
                                FROM sqlitestudio_temp_table;

DROP TABLE sqlitestudio_temp_table;

CREATE INDEX measurepersection_measure_id ON MeasurePerSection (
    "measure_id"
);

CREATE INDEX measurepersection_section_id ON MeasurePerSection (
    "section_id"
);

PRAGMA foreign_keys = 1;

-- MeasureResult
PRAGMA foreign_keys = 0;

CREATE TABLE sqlitestudio_temp_table AS SELECT *
                                          FROM MeasureResult;

DROP TABLE MeasureResult;

CREATE TABLE MeasureResult (
    id                     INTEGER NOT NULL
                                   PRIMARY KEY,
    measure_per_section_id INTEGER NOT NULL,
    FOREIGN KEY (
        measure_per_section_id
    )
    REFERENCES MeasurePerSection (id) ON DELETE CASCADE
);

INSERT INTO MeasureResult (
                              id,
                              measure_per_section_id
                          )
                          SELECT id,
                                 measure_per_section_id
                            FROM sqlitestudio_temp_table;

DROP TABLE sqlitestudio_temp_table;

CREATE INDEX measureresult_measure_per_section_id ON MeasureResult (
    "measure_per_section_id"
);

PRAGMA foreign_keys = 1;

-- MeasureResultMechanism
PRAGMA foreign_keys = 0;

CREATE TABLE sqlitestudio_temp_table AS SELECT *
                                          FROM MeasureResultMechanism;

DROP TABLE MeasureResultMechanism;

CREATE TABLE MeasureResultMechanism (
    id                       INTEGER NOT NULL
                                     PRIMARY KEY,
    measure_result_id        INTEGER NOT NULL,
    mechanism_per_section_id INTEGER NOT NULL,
    beta                     REAL    NOT NULL,
    time                     INTEGER NOT NULL,
    FOREIGN KEY (
        measure_result_id
    )
    REFERENCES MeasureResult (id) ON DELETE CASCADE,
    FOREIGN KEY (
        mechanism_per_section_id
    )
    REFERENCES MechanismPerSection (id) ON DELETE CASCADE
);

INSERT INTO MeasureResultMechanism (
                                       id,
                                       measure_result_id,
                                       mechanism_per_section_id,
                                       beta,
                                       time
                                   )
                                   SELECT id,
                                          measure_result_id,
                                          mechanism_per_section_id,
                                          beta,
                                          time
                                     FROM sqlitestudio_temp_table;

DROP TABLE sqlitestudio_temp_table;

CREATE INDEX measureresultmechanism_measure_result_id ON MeasureResultMechanism (
    "measure_result_id"
);

CREATE INDEX measureresultmechanism_mechanism_per_section_id ON MeasureResultMechanism (
    "mechanism_per_section_id"
);

PRAGMA foreign_keys = 1;

-- MeasureResultSection
PRAGMA foreign_keys = 0;

CREATE TABLE sqlitestudio_temp_table AS SELECT *
                                          FROM MeasureResultSection;

DROP TABLE MeasureResultSection;

CREATE TABLE MeasureResultSection (
    id                INTEGER NOT NULL
                              PRIMARY KEY,
    measure_result_id INTEGER NOT NULL,
    beta              REAL    NOT NULL,
    time              INTEGER NOT NULL,
    cost              REAL    NOT NULL,
    FOREIGN KEY (
        measure_result_id
    )
    REFERENCES MeasureResult (id) ON DELETE CASCADE
);

INSERT INTO MeasureResultSection (
                                     id,
                                     measure_result_id,
                                     beta,
                                     time,
                                     cost
                                 )
                                 SELECT id,
                                        measure_result_id,
                                        beta,
                                        time,
                                        cost
                                   FROM sqlitestudio_temp_table;

DROP TABLE sqlitestudio_temp_table;

CREATE INDEX measureresultsection_measure_result_id ON MeasureResultSection (
    "measure_result_id"
);

PRAGMA foreign_keys = 1;

-- MeasureResultParameter
PRAGMA foreign_keys = 0;

CREATE TABLE sqlitestudio_temp_table AS SELECT *
                                          FROM MeasureResultParameter;

DROP TABLE MeasureResultParameter;

CREATE TABLE MeasureResultParameter (
    id                INTEGER       NOT NULL
                                    PRIMARY KEY,
    name              VARCHAR (128) NOT NULL,
    value             REAL          NOT NULL,
    measure_result_id INTEGER       NOT NULL,
    FOREIGN KEY (
        measure_result_id
    )
    REFERENCES MeasureResult (id) ON DELETE CASCADE
);

INSERT INTO MeasureResultParameter (
                                       id,
                                       name,
                                       value,
                                       measure_result_id
                                   )
                                   SELECT id,
                                          name,
                                          value,
                                          measure_result_id
                                     FROM sqlitestudio_temp_table;

DROP TABLE sqlitestudio_temp_table;

CREATE INDEX measureresultparameter_measure_result_id ON MeasureResultParameter (
    "measure_result_id"
);

PRAGMA foreign_keys = 1;

-- OptimizationRun
PRAGMA foreign_keys = 0;

CREATE TABLE sqlitestudio_temp_table AS SELECT *
                                          FROM OptimizationRun;

DROP TABLE OptimizationRun;

CREATE TABLE OptimizationRun (
    id                   INTEGER       NOT NULL
                                       PRIMARY KEY,
    name                 VARCHAR (128) NOT NULL,
    discount_rate        REAL          NOT NULL,
    optimization_type_id INTEGER       NOT NULL,
    FOREIGN KEY (
        optimization_type_id
    )
    REFERENCES OptimizationType (id) ON DELETE CASCADE
);

INSERT INTO OptimizationRun (
                                id,
                                name,
                                discount_rate,
                                optimization_type_id
                            )
                            SELECT id,
                                   name,
                                   discount_rate,
                                   optimization_type_id
                              FROM sqlitestudio_temp_table;

DROP TABLE sqlitestudio_temp_table;

CREATE UNIQUE INDEX optimizationrun_name ON OptimizationRun (
    "name"
);

CREATE INDEX optimizationrun_optimization_type_id ON OptimizationRun (
    "optimization_type_id"
);

PRAGMA foreign_keys = 1;

-- OptimizationSelectedMeasure
PRAGMA foreign_keys = 0;

CREATE TABLE sqlitestudio_temp_table AS SELECT *
                                          FROM OptimizationSelectedMeasure;

DROP TABLE OptimizationSelectedMeasure;

CREATE TABLE OptimizationSelectedMeasure (
    id                  INTEGER NOT NULL
                                PRIMARY KEY,
    optimization_run_id INTEGER NOT NULL,
    measure_result_id   INTEGER NOT NULL,
    investment_year     INTEGER NOT NULL,
    FOREIGN KEY (
        optimization_run_id
    )
    REFERENCES OptimizationRun (id) ON DELETE CASCADE,
    FOREIGN KEY (
        measure_result_id
    )
    REFERENCES MeasureResult (id) ON DELETE CASCADE
);

INSERT INTO OptimizationSelectedMeasure (
                                            id,
                                            optimization_run_id,
                                            measure_result_id,
                                            investment_year
                                        )
                                        SELECT id,
                                               optimization_run_id,
                                               measure_result_id,
                                               investment_year
                                          FROM sqlitestudio_temp_table;

DROP TABLE sqlitestudio_temp_table;

CREATE INDEX optimizationselectedmeasure_measure_result_id ON OptimizationSelectedMeasure (
    "measure_result_id"
);

CREATE INDEX optimizationselectedmeasure_optimization_run_id ON OptimizationSelectedMeasure (
    "optimization_run_id"
);

PRAGMA foreign_keys = 1;

-- OptimizationStep

PRAGMA foreign_keys = 0;

CREATE TABLE sqlitestudio_temp_table AS SELECT *
                                          FROM OptimizationStep;

DROP TABLE OptimizationStep;

CREATE TABLE OptimizationStep (
    id                               INTEGER NOT NULL
                                             PRIMARY KEY,
    optimization_selected_measure_id INTEGER NOT NULL,
    step_number                      INTEGER NOT NULL,
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

-- StandardMeasure
PRAGMA foreign_keys = 0;

CREATE TABLE sqlitestudio_temp_table AS SELECT *
                                          FROM StandardMeasure;

DROP TABLE StandardMeasure;

CREATE TABLE StandardMeasure (
    id                                INTEGER       NOT NULL
                                                    PRIMARY KEY,
    measure_id                        INTEGER       NOT NULL,
    max_inward_reinforcement          INTEGER       NOT NULL,
    max_outward_reinforcement         INTEGER       NOT NULL,
    direction                         VARCHAR (128) NOT NULL,
    crest_step                        REAL          NOT NULL,
    max_crest_increase                REAL          NOT NULL,
    stability_screen                  INTEGER       NOT NULL,
    prob_of_solution_failure          REAL          NOT NULL,
    failure_probability_with_solution REAL          NOT NULL,
    stability_screen_s_f_increase     REAL          NOT NULL,
    transition_level_increase_step    REAL          NOT NULL,
    max_pf_factor_block               REAL          NOT NULL,
    n_steps_block                     INTEGER       NOT NULL,
    FOREIGN KEY (
        measure_id
    )
    REFERENCES Measure (id) ON DELETE CASCADE
);

INSERT INTO StandardMeasure (
                                id,
                                measure_id,
                                max_inward_reinforcement,
                                max_outward_reinforcement,
                                direction,
                                crest_step,
                                max_crest_increase,
                                stability_screen,
                                prob_of_solution_failure,
                                failure_probability_with_solution,
                                stability_screen_s_f_increase,
                                transition_level_increase_step,
                                max_pf_factor_block,
                                n_steps_block
                            )
                            SELECT id,
                                   measure_id,
                                   max_inward_reinforcement,
                                   max_outward_reinforcement,
                                   direction,
                                   crest_step,
                                   max_crest_increase,
                                   stability_screen,
                                   prob_of_solution_failure,
                                   failure_probability_with_solution,
                                   stability_screen_s_f_increase,
                                   transition_level_increase_step,
                                   max_pf_factor_block,
                                   n_steps_block
                              FROM sqlitestudio_temp_table;

DROP TABLE sqlitestudio_temp_table;

CREATE UNIQUE INDEX standardmeasure_measure_id ON StandardMeasure (
    "measure_id"
);

PRAGMA foreign_keys = 1;

-- Changes required from VRTOOL-514
DROP TABLE CustomMeasureParameter;

DROP TABLE IF EXISTS CustomMeasurePerMeasurePerSection;

-- Renaming of `CustomMeasure` to `CustomMeasureDetail`
PRAGMA foreign_keys = 0;

CREATE TABLE CustomMeasureDetail (
    id           INTEGER NOT NULL
                         PRIMARY KEY,
    measure_id   INTEGER NOT NULL,
    mechanism_id INTEGER NOT NULL,
    cost         REAL,
    beta         REAL,
    year         INTEGER NOT NULL,
    FOREIGN KEY (
        measure_id
    )
    REFERENCES Measure (id) ON DELETE CASCADE,
    FOREIGN KEY (
        mechanism_id
    )
    REFERENCES Mechanism (id) ON DELETE CASCADE
);

INSERT INTO CustomMeasureDetail (
                                    id,
                                    measure_id,
                                    mechanism_id,
                                    cost,
                                    beta,
                                    year
                                )
                                SELECT id,
                                       measure_id,
                                       mechanism_id,
                                       cost,
                                       beta,
                                       year
                                  FROM CustomMeasure;

CREATE TABLE CustomMeasureDetailPerSection (
    id                     INTEGER NOT NULL
                                   PRIMARY KEY,
    measure_per_section_id INTEGER NOT NULL,
    custom_measure_detail_id      INTEGER NOT NULL,
    FOREIGN KEY (
        measure_per_section_id
    )
    REFERENCES MeasurePerSection (id) ON DELETE CASCADE,
    FOREIGN KEY (
        custom_measure_detail_id
    )
    REFERENCES CustomMeasureDetail (id) ON DELETE CASCADE
);

DROP TABLE CustomMeasure;

CREATE INDEX custommeasure_measure_id ON CustomMeasureDetail (
    "measure_id"
);

CREATE INDEX custommeasure_mechanism_id ON CustomMeasureDetail (
    "mechanism_id"
);

PRAGMA foreign_keys = 1;



-- General pragma changes
PRAGMA journal_mode = "WAL";
PRAGMA cache_size = -64000;
PRAGMA foreign_keys = 1;
PRAGMA ignore_check_constraints = 0;
PRAGMA synchronous = 0;
