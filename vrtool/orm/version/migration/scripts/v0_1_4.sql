-- MIGRATION FILE
DROP TABLE IF EXISTS sqlitestudio_temp_table;

-- VRTOOL-546 - Adding `n_revetment` and `n_overflow` to `DikeTrajectInfo`.
PRAGMA foreign_keys = 0;

CREATE TABLE sqlitestudio_temp_table AS SELECT *
                                          FROM DikeTrajectInfo;

DROP TABLE DikeTrajectInfo;

CREATE TABLE DikeTrajectInfo (
    id                    INTEGER       NOT NULL
                                        PRIMARY KEY,
    traject_name          VARCHAR (128) NOT NULL,
    omega_piping          REAL          NOT NULL,
    omega_stability_inner REAL          NOT NULL,
    omega_overflow        REAL          NOT NULL,
    a_piping              REAL,
    b_piping              REAL          NOT NULL,
    a_stability_inner     REAL          NOT NULL,
    b_stability_inner     REAL          NOT NULL,
    beta_max              REAL,
    p_max                 REAL,
    flood_damage          REAL,
    traject_length        REAL,
    n_revetment           INTEGER       DEFAULT (3),
    n_overflow                          DEFAULT (1) 
);

INSERT INTO DikeTrajectInfo (
                                id,
                                traject_name,
                                omega_piping,
                                omega_stability_inner,
                                omega_overflow,
                                a_piping,
                                b_piping,
                                a_stability_inner,
                                b_stability_inner,
                                beta_max,
                                p_max,
                                flood_damage,
                                traject_length
                            )
                            SELECT id,
                                   traject_name,
                                   omega_piping,
                                   omega_stability_inner,
                                   omega_overflow,
                                   a_piping,
                                   b_piping,
                                   a_stability_inner,
                                   b_stability_inner,
                                   beta_max,
                                   p_max,
                                   flood_damage,
                                   traject_length
                              FROM sqlitestudio_temp_table;

DROP TABLE sqlitestudio_temp_table;

PRAGMA foreign_keys = 1;
