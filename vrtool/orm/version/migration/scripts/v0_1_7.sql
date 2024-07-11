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
