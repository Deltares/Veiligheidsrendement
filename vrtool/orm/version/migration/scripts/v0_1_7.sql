DROP TABLE IF EXISTS sqlitestudio_temp_table;

-- VRTOOL-544 Remove columns from `StandardMeasure`:
-- `rob_of_solution_failure`, `failure_probability_with_solution`, `stability_screen_s_f_increase`
-- CustomMeasureDetail

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
    transition_level_increase_step    REAL          NOT NULL,
    max_pf_factor_block               REAL          NOT NULL,
    n_steps_block                     INTEGER       NOT NULL,
    piping_reduction_factor           REAL,
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
                                transition_level_increase_step,
                                max_pf_factor_block,
                                n_steps_block,
                                piping_reduction_factor
                            )
                            SELECT id,
                                   measure_id,
                                   max_inward_reinforcement,
                                   max_outward_reinforcement,
                                   direction,
                                   crest_step,
                                   max_crest_increase,
                                   stability_screen,
                                   transition_level_increase_step,
                                   max_pf_factor_block,
                                   n_steps_block,
                                   (1 / prob_of_solution_failure)
                              FROM sqlitestudio_temp_table;

-- Replace any invalid values when `prob_of_solution_failure` was -999
UPDATE StandardMeasure
SET   piping_reduction_factor = NULL
WHERE piping_reduction_factor <= 0;

DROP TABLE sqlitestudio_temp_table;

CREATE UNIQUE INDEX standardmeasure_measure_id ON StandardMeasure (
    "measure_id"
);

PRAGMA foreign_keys = 1;