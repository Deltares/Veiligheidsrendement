DROP TABLE IF EXISTS sqlitestudio_temp_table;

-- VRTOOL-542 Migration of tables to include `on_delete="CASCADE"` for all foreign keys.
-- AssessmentMechanismResult
PRAGMA foreign_keys = 0;

CREATE TABLE sqlitestudio_temp_table AS SELECT *
                                          FROM AssessmentMechanismResult;

DROP TABLE AssessmentMechanismResult;

CREATE TABLE AssessmentMechanismResult (
    id                       INTEGER NOT NULL
                                     PRIMARY KEY,
    beta                     REAL    NOT NULL,
    time                     INTEGER NOT NULL,
    mechanism_per_section_id INTEGER NOT NULL,
    FOREIGN KEY (
        mechanism_per_section_id
    )
    REFERENCES MechanismPerSection (id) ON DELETE CASCADE
);

INSERT INTO AssessmentMechanismResult (
                                          id,
                                          beta,
                                          time,
                                          mechanism_per_section_id
                                      )
                                      SELECT id,
                                             beta,
                                             time,
                                             mechanism_per_section_id
                                        FROM sqlitestudio_temp_table;

DROP TABLE sqlitestudio_temp_table;

CREATE INDEX assessmentmechanismresult_mechanism_per_section_id ON AssessmentMechanismResult (
    "mechanism_per_section_id"
);

PRAGMA foreign_keys = 1;

-- AssessmentSectionResult
PRAGMA foreign_keys = 0;

CREATE TABLE sqlitestudio_temp_table AS SELECT *
                                          FROM AssessmentSectionResult;

DROP TABLE AssessmentSectionResult;

CREATE TABLE AssessmentSectionResult (
    id              INTEGER NOT NULL
                            PRIMARY KEY,
    beta            REAL    NOT NULL,
    time            INTEGER NOT NULL,
    section_data_id INTEGER NOT NULL,
    FOREIGN KEY (
        section_data_id
    )
    REFERENCES SectionData (id) ON DELETE CASCADE
);

INSERT INTO AssessmentSectionResult (
                                        id,
                                        beta,
                                        time,
                                        section_data_id
                                    )
                                    SELECT id,
                                           beta,
                                           time,
                                           section_data_id
                                      FROM sqlitestudio_temp_table;

DROP TABLE sqlitestudio_temp_table;

CREATE INDEX assessmentsectionresult_section_data_id ON AssessmentSectionResult (
    "section_data_id"
);

PRAGMA foreign_keys = 1;

-- BlockRevetmentRelation
PRAGMA foreign_keys = 0;

CREATE TABLE sqlitestudio_temp_table AS SELECT *
                                          FROM BlockRevetmentRelation;

DROP TABLE BlockRevetmentRelation;

CREATE TABLE BlockRevetmentRelation (
    id                  INTEGER NOT NULL
                                PRIMARY KEY,
    slope_part_id       INTEGER NOT NULL,
    year                INTEGER NOT NULL,
    top_layer_thickness REAL    NOT NULL,
    beta                REAL    NOT NULL,
    FOREIGN KEY (
        slope_part_id
    )
    REFERENCES SlopePart (id) ON DELETE CASCADE
);

INSERT INTO BlockRevetmentRelation (
                                       id,
                                       slope_part_id,
                                       year,
                                       top_layer_thickness,
                                       beta
                                   )
                                   SELECT id,
                                          slope_part_id,
                                          year,
                                          top_layer_thickness,
                                          beta
                                     FROM sqlitestudio_temp_table;

DROP TABLE sqlitestudio_temp_table;

CREATE INDEX blockrevetmentrelation_slope_part_id ON BlockRevetmentRelation (
    "slope_part_id"
);

PRAGMA foreign_keys = 1;

--Buildings
PRAGMA foreign_keys = 0;

CREATE TABLE sqlitestudio_temp_table AS SELECT *
                                          FROM Buildings;

DROP TABLE Buildings;

CREATE TABLE Buildings (
    id                  INTEGER NOT NULL
                                PRIMARY KEY,
    section_data_id     INTEGER NOT NULL,
    distance_from_toe   REAL    NOT NULL,
    number_of_buildings INTEGER NOT NULL,
    FOREIGN KEY (
        section_data_id
    )
    REFERENCES SectionData (id) ON DELETE CASCADE
);

INSERT INTO Buildings (
                          id,
                          section_data_id,
                          distance_from_toe,
                          number_of_buildings
                      )
                      SELECT id,
                             section_data_id,
                             distance_from_toe,
                             number_of_buildings
                        FROM sqlitestudio_temp_table;

DROP TABLE sqlitestudio_temp_table;

CREATE INDEX buildings_section_data_id ON Buildings (
    "section_data_id"
);

PRAGMA foreign_keys = 1;

--ComputationScenario
PRAGMA foreign_keys = 0;

CREATE TABLE sqlitestudio_temp_table AS SELECT *
                                          FROM ComputationScenario;

DROP TABLE ComputationScenario;

CREATE TABLE ComputationScenario (
    id                       INTEGER       NOT NULL
                                           PRIMARY KEY,
    mechanism_per_section_id INTEGER       NOT NULL,
    computation_type_id      INTEGER       NOT NULL,
    computation_name         VARCHAR (128) NOT NULL,
    scenario_name            VARCHAR (128) NOT NULL,
    scenario_probability     REAL,
    probability_of_failure   REAL          NOT NULL,
    FOREIGN KEY (
        mechanism_per_section_id
    )
    REFERENCES MechanismPerSection (id) ON DELETE CASCADE,
    FOREIGN KEY (
        computation_type_id
    )
    REFERENCES ComputationType (id) ON DELETE CASCADE
);

INSERT INTO ComputationScenario (
                                    id,
                                    mechanism_per_section_id,
                                    computation_type_id,
                                    computation_name,
                                    scenario_name,
                                    scenario_probability,
                                    probability_of_failure
                                )
                                SELECT id,
                                       mechanism_per_section_id,
                                       computation_type_id,
                                       computation_name,
                                       scenario_name,
                                       scenario_probability,
                                       probability_of_failure
                                  FROM sqlitestudio_temp_table;

DROP TABLE sqlitestudio_temp_table;

CREATE INDEX computationscenario_computation_type_id ON ComputationScenario (
    "computation_type_id"
);

CREATE INDEX computationscenario_mechanism_per_section_id ON ComputationScenario (
    "mechanism_per_section_id"
);

PRAGMA foreign_keys = 1;

-- ComputationScenarioParameter
PRAGMA foreign_keys = 0;

CREATE TABLE sqlitestudio_temp_table AS SELECT *
                                          FROM ComputationScenarioParameter;

DROP TABLE ComputationScenarioParameter;

CREATE TABLE ComputationScenarioParameter (
    id                      INTEGER       NOT NULL
                                          PRIMARY KEY,
    computation_scenario_id INTEGER       NOT NULL,
    parameter               VARCHAR (128) NOT NULL,
    value                   REAL          NOT NULL,
    FOREIGN KEY (
        computation_scenario_id
    )
    REFERENCES ComputationScenario (id) ON DELETE CASCADE
);

INSERT INTO ComputationScenarioParameter (
                                             id,
                                             computation_scenario_id,
                                             parameter,
                                             value
                                         )
                                         SELECT id,
                                                computation_scenario_id,
                                                parameter,
                                                value
                                           FROM sqlitestudio_temp_table;

DROP TABLE sqlitestudio_temp_table;

CREATE INDEX computationscenarioparameter_computation_scenario_id ON ComputationScenarioParameter (
    "computation_scenario_id"
);

PRAGMA foreign_keys = 1;

-- GrassRevetmentRelation
PRAGMA foreign_keys = 0;

CREATE TABLE sqlitestudio_temp_table AS SELECT *
                                          FROM GrassRevetmentRelation;

DROP TABLE GrassRevetmentRelation;

CREATE TABLE GrassRevetmentRelation (
    id                      INTEGER NOT NULL
                                    PRIMARY KEY,
    computation_scenario_id INTEGER NOT NULL,
    year                    INTEGER NOT NULL,
    transition_level        REAL    NOT NULL,
    beta                    REAL    NOT NULL,
    FOREIGN KEY (
        computation_scenario_id
    )
    REFERENCES ComputationScenario (id) ON DELETE CASCADE
);

INSERT INTO GrassRevetmentRelation (
                                       id,
                                       computation_scenario_id,
                                       year,
                                       transition_level,
                                       beta
                                   )
                                   SELECT id,
                                          computation_scenario_id,
                                          year,
                                          transition_level,
                                          beta
                                     FROM sqlitestudio_temp_table;

DROP TABLE sqlitestudio_temp_table;

CREATE INDEX grassrevetmentrelation_computation_scenario_id ON GrassRevetmentRelation (
    "computation_scenario_id"
);

PRAGMA foreign_keys = 1;

-- MechanismPerSection
PRAGMA foreign_keys = 0;

CREATE TABLE sqlitestudio_temp_table AS SELECT *
                                          FROM MechanismPerSection;

DROP TABLE MechanismPerSection;

CREATE TABLE MechanismPerSection (
    id           INTEGER NOT NULL
                         PRIMARY KEY,
    section_id   INTEGER NOT NULL,
    mechanism_id INTEGER NOT NULL,
    FOREIGN KEY (
        section_id
    )
    REFERENCES SectionData (id) ON DELETE CASCADE,
    FOREIGN KEY (
        mechanism_id
    )
    REFERENCES Mechanism (id) ON DELETE CASCADE
);

INSERT INTO MechanismPerSection (
                                    id,
                                    section_id,
                                    mechanism_id
                                )
                                SELECT id,
                                       section_id,
                                       mechanism_id
                                  FROM sqlitestudio_temp_table;

DROP TABLE sqlitestudio_temp_table;

CREATE INDEX mechanismpersection_mechanism_id ON MechanismPerSection (
    "mechanism_id"
);

CREATE INDEX mechanismpersection_section_id ON MechanismPerSection (
    "section_id"
);

PRAGMA foreign_keys = 1;

-- MechanismTable
PRAGMA foreign_keys = 0;

CREATE TABLE sqlitestudio_temp_table AS SELECT *
                                          FROM MechanismTable;

DROP TABLE MechanismTable;

CREATE TABLE MechanismTable (
    id                      INTEGER NOT NULL
                                    PRIMARY KEY,
    computation_scenario_id INTEGER NOT NULL,
    year                    INTEGER NOT NULL,
    value                   REAL    NOT NULL,
    beta                    REAL    NOT NULL,
    FOREIGN KEY (
        computation_scenario_id
    )
    REFERENCES ComputationScenario (id) ON DELETE CASCADE
);

INSERT INTO MechanismTable (
                               id,
                               computation_scenario_id,
                               year,
                               value,
                               beta
                           )
                           SELECT id,
                                  computation_scenario_id,
                                  year,
                                  value,
                                  beta
                             FROM sqlitestudio_temp_table;

DROP TABLE sqlitestudio_temp_table;

CREATE INDEX mechanismtable_computation_scenario_id ON MechanismTable (
    "computation_scenario_id"
);

PRAGMA foreign_keys = 1;

-- ProfilePoint
PRAGMA foreign_keys = 0;

CREATE TABLE sqlitestudio_temp_table AS SELECT *
                                          FROM ProfilePoint;

DROP TABLE ProfilePoint;

CREATE TABLE ProfilePoint (
    id                    INTEGER NOT NULL
                                  PRIMARY KEY,
    profile_point_type_id INTEGER NOT NULL,
    section_data_id       INTEGER NOT NULL,
    x_coordinate          REAL    NOT NULL,
    y_coordinate          REAL    NOT NULL,
    FOREIGN KEY (
        profile_point_type_id
    )
    REFERENCES CharacteristicPointType (id) ON DELETE CASCADE,
    FOREIGN KEY (
        section_data_id
    )
    REFERENCES SectionData (id) ON DELETE CASCADE
);

INSERT INTO ProfilePoint (
                             id,
                             profile_point_type_id,
                             section_data_id,
                             x_coordinate,
                             y_coordinate
                         )
                         SELECT id,
                                profile_point_type_id,
                                section_data_id,
                                x_coordinate,
                                y_coordinate
                           FROM sqlitestudio_temp_table;

DROP TABLE sqlitestudio_temp_table;

CREATE INDEX profilepoint_profile_point_type_id ON ProfilePoint (
    "profile_point_type_id"
);

CREATE INDEX profilepoint_section_data_id ON ProfilePoint (
    "section_data_id"
);

PRAGMA foreign_keys = 1;

-- SectionData
PRAGMA foreign_keys = 0;

CREATE TABLE sqlitestudio_temp_table AS SELECT *
                                          FROM SectionData;

DROP TABLE SectionData;

CREATE TABLE SectionData (
    id                    INTEGER       NOT NULL
                                        PRIMARY KEY,
    dike_traject_id       INTEGER       NOT NULL,
    section_name          VARCHAR (128) NOT NULL,
    dijkpaal_start        VARCHAR (128),
    dijkpaal_end          VARCHAR (128),
    meas_start            REAL          NOT NULL,
    meas_end              REAL          NOT NULL,
    section_length        REAL          NOT NULL,
    in_analysis           INTEGER       NOT NULL,
    crest_height          REAL          NOT NULL,
    annual_crest_decline  REAL          NOT NULL,
    cover_layer_thickness REAL          NOT NULL,
    pleistocene_level     REAL          NOT NULL,
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
                            pleistocene_level
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
                               pleistocene_level
                          FROM sqlitestudio_temp_table;

DROP TABLE sqlitestudio_temp_table;

CREATE INDEX sectiondata_dike_traject_id ON SectionData (
    "dike_traject_id"
);

CREATE UNIQUE INDEX sectiondata_section_name ON SectionData (
    "section_name"
);

PRAGMA foreign_keys = 1;

-- SlopePart
PRAGMA foreign_keys = 0;

CREATE TABLE sqlitestudio_temp_table AS SELECT *
                                          FROM SlopePart;

DROP TABLE SlopePart;

CREATE TABLE SlopePart (
    id                      INTEGER NOT NULL
                                    PRIMARY KEY,
    computation_scenario_id INTEGER NOT NULL,
    begin_part              REAL    NOT NULL,
    end_part                REAL    NOT NULL,
    top_layer_type          REAL    NOT NULL,
    top_layer_thickness     REAL,
    tan_alpha               REAL    NOT NULL,
    FOREIGN KEY (
        computation_scenario_id
    )
    REFERENCES ComputationScenario (id) ON DELETE CASCADE
);

INSERT INTO SlopePart (
                          id,
                          computation_scenario_id,
                          begin_part,
                          end_part,
                          top_layer_type,
                          top_layer_thickness,
                          tan_alpha
                      )
                      SELECT id,
                             computation_scenario_id,
                             begin_part,
                             end_part,
                             top_layer_type,
                             top_layer_thickness,
                             tan_alpha
                        FROM sqlitestudio_temp_table;

DROP TABLE sqlitestudio_temp_table;

CREATE INDEX slopepart_computation_scenario_id ON SlopePart (
    "computation_scenario_id"
);

PRAGMA foreign_keys = 1;

-- SupportingFile
PRAGMA foreign_keys = 0;

CREATE TABLE sqlitestudio_temp_table AS SELECT *
                                          FROM SupportingFile;

DROP TABLE SupportingFile;

CREATE TABLE SupportingFile (
    id                      INTEGER       NOT NULL
                                          PRIMARY KEY,
    computation_scenario_id INTEGER       NOT NULL,
    filename                VARCHAR (128) NOT NULL,
    FOREIGN KEY (
        computation_scenario_id
    )
    REFERENCES ComputationScenario (id) ON DELETE CASCADE
);

INSERT INTO SupportingFile (
                               id,
                               computation_scenario_id,
                               filename
                           )
                           SELECT id,
                                  computation_scenario_id,
                                  filename
                             FROM sqlitestudio_temp_table;

DROP TABLE sqlitestudio_temp_table;

CREATE INDEX supportingfile_computation_scenario_id ON SupportingFile (
    "computation_scenario_id"
);

PRAGMA foreign_keys = 1;

-- WaterLevelData
PRAGMA foreign_keys = 0;

CREATE TABLE sqlitestudio_temp_table AS SELECT *
                                          FROM WaterlevelData;

DROP TABLE WaterlevelData;

CREATE TABLE WaterlevelData (
    id                      INTEGER NOT NULL
                                    PRIMARY KEY,
    section_data_id         INTEGER NOT NULL,
    water_level_location_id INTEGER,
    year                    INTEGER NOT NULL,
    water_level             REAL    NOT NULL,
    beta                    REAL    NOT NULL,
    FOREIGN KEY (
        section_data_id
    )
    REFERENCES SectionData (id) ON DELETE CASCADE
);

INSERT INTO WaterlevelData (
                               id,
                               section_data_id,
                               water_level_location_id,
                               year,
                               water_level,
                               beta
                           )
                           SELECT id,
                                  section_data_id,
                                  water_level_location_id,
                                  year,
                                  water_level,
                                  beta
                             FROM sqlitestudio_temp_table;

DROP TABLE sqlitestudio_temp_table;

CREATE INDEX waterleveldata_section_data_id ON WaterlevelData (
    "section_data_id"
);

PRAGMA foreign_keys = 1;
