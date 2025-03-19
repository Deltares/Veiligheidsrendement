DROP TABLE IF EXISTS sqlitestudio_temp_table;

-- VRTOOL-188  Added columns to `SectionData`: 
-- `sensitive_fraction_piping` and 
-- `sensitive_fraction_stability_inner`
PRAGMA foreign_keys = 0;

CREATE TABLE sqlitestudio_temp_table AS SELECT *
                                          FROM SectionData;

DROP TABLE SectionData;

CREATE TABLE SectionData (
    id                                 INTEGER       NOT NULL
                                                     PRIMARY KEY,
    dike_traject_id                    INTEGER       NOT NULL,
    section_name                       VARCHAR (128) NOT NULL,
    dijkpaal_start                     VARCHAR (128),
    dijkpaal_end                       VARCHAR (128),
    meas_start                         REAL          NOT NULL,
    meas_end                           REAL          NOT NULL,
    section_length                     REAL          NOT NULL,
    in_analysis                        INTEGER       NOT NULL,
    crest_height                       REAL          NOT NULL,
    annual_crest_decline               REAL          NOT NULL,
    cover_layer_thickness              REAL          NOT NULL
                                                     DEFAULT (7),
    pleistocene_level                  REAL          NOT NULL
                                                     DEFAULT (25),
    flood_damage                       REAL,
    sensitive_fraction_piping          REAL          DEFAULT (0),
    sensitive_fraction_stability_inner REAL          DEFAULT (0),
    FOREIGN KEY (
        dike_traject_id
    )
    REFERENCES DikeTrajectInfo (id) ON DELETE CASCADE
);

INSERT INTO SectionData (
                            id,
                            dike_traject_id,
                            section_name,
                            dijkpaal_start,
                            dijkpaal_end,
                            meas_start,
                            meas_end,
                            section_length,
                            in_analysis,
                            crest_height,
                            annual_crest_decline,
                            cover_layer_thickness,
                            pleistocene_level,
                            flood_damage
                        )
                        SELECT id,
                               dike_traject_id,
                               section_name,
                               dijkpaal_start,
                               dijkpaal_end,
                               meas_start,
                               meas_end,
                               section_length,
                               in_analysis,
                               crest_height,
                               annual_crest_decline,
                               cover_layer_thickness,
                               pleistocene_level,
                               flood_damage
                          FROM sqlitestudio_temp_table;

DROP TABLE sqlitestudio_temp_table;

CREATE INDEX sectiondata_dike_traject_id ON SectionData (
    "dike_traject_id"
);

CREATE UNIQUE INDEX sectiondata_section_name ON SectionData (
    "section_name"
);

PRAGMA foreign_keys = 1;
