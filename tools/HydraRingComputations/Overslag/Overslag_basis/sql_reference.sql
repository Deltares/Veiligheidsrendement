--! Basic input file for GEKB computation
-- KEYS TO BE FILLED: 'LocationName', HLCDID, ORIENTATION, TimeIntegration, Hmin, Hmax, Hstep, data from profile
DELETE FROM [HydraulicModels];
INSERT INTO [HydraulicModels] VALUES (1, 1, 'WTI 2017');

DELETE FROM [Sections];
INSERT INTO [Sections] VALUES (1, 1, 1, 'LocationName', 'LocationName', -999,-999,-999,-999, HLCDID, HLCDID, 100, ORIENTATION, 0);

DELETE FROM [SectionCalculationSchemes];
INSERT INTO [SectionCalculationSchemes] VALUES (1, 101, 1);

DELETE FROM [DesignTables];
INSERT INTO [DesignTables] VALUES (1, 101, 1, 1, 3, 1, NULL, Hmin, Hmax, Hstep, 0, 0, 0);

DELETE FROM [PreprocessorSettings];

DELETE FROM [Numerics];
INSERT INTO [Numerics] VALUES (1, 101, 1, 1, 102, 12, 4, 150, 0.15000001, 0.005, 0.005, 0.005, 2, 3, 10000, 40000, 0.1, -6, 6, 25);
INSERT INTO [Numerics] VALUES (1, 101, 1, 1, 103, 12, 4, 150, 0.15000001, 0.005, 0.005, 0.005, 2, 3, 10000, 40000, 0.1, -6, 6, 25);

DELETE FROM [VariableDatas];
INSERT INTO [VariableDatas] VALUES (1, 101, 1, 1, 1, KRUINHOOGTE, 0, 0, NULL, NULL, NULL, 1, 0, 300);
INSERT INTO [VariableDatas] VALUES (1, 101, 1, 1, 8, 1, 0, 0, NULL, NULL, NULL, 1, 0, 300);
INSERT INTO [VariableDatas] VALUES (1, 101, 1, 1, 10, 0, 19, 4.75, 0.5, 0, 99, 1, 0, 300);
INSERT INTO [VariableDatas] VALUES (1, 101, 1, 1, 11, 0, 19, 2.6, 0.35, 0, 99, 1, 0, 300);
INSERT INTO [VariableDatas] VALUES (1, 101, 1, 1, 12, 1, 0, 0, NULL, NULL, NULL, 1, 0, 300);
INSERT INTO [VariableDatas] VALUES (1, 101, 1, 1, 17, 0, 4, MU_QC, SIGMA_QC, NULL, NULL, 1, 0, 300);
INSERT INTO [VariableDatas] VALUES (1, 101, 1, 1, 120, 0, 19, 1, 0.07, 0, 99, 1, 0, 300);
INSERT INTO [VariableDatas] VALUES (1, 101, 1, 1, 123, 0, 19, 0.92, 0.24, 0, 99, 1, 0, 300);

DELETE FROM [SectionFaultTreeModels];
INSERT INTO [SectionFaultTreeModels] VALUES (1, 101, 1, 1, 1017);

DELETE FROM [SectionSubMechanismModels];
INSERT INTO [SectionSubMechanismModels] VALUES (1, 1, 1, 102, 94);
INSERT INTO [SectionSubMechanismModels] VALUES (1, 1, 1, 103, 95);

DELETE FROM [Fetches];

DELETE FROM [AreaPoints];

DELETE FROM [PresentationSections];

--Below is only the part that is used for the dike profile: meaning all data to be read from the prfl 
--(Forelands, ForelandModels, Profiles, CalculationProfiles, Breakwaters)

DELETE FROM [CalculationProfiles];
INSERTCALCULATIONPROFILE
--Example line: INSERT INTO [CalculationProfiles] VALUES (1, 1, -17, 5.13, 1);

DELETE FROM [Profiles];
INSERTPROFILES
--Example line: INSERT INTO [Profiles] VALUES (1, 1, -17, 5.13);

DELETE FROM [ForelandModels];
INSERT INTO [ForelandModels] VALUES (1, 101, 3);

DELETE FROM [Forelands];
INSERTFORELANDGEOMETRY
--Example line: INSERT INTO [Forelands] VALUES (1, 1, -68, 2.56);

DELETE FROM [ProbabilityAlternatives];

DELETE FROM [SetUpHeights];

DELETE FROM [CalcWindDirections];

DELETE FROM [Swells];

DELETE FROM [WaveReductions];

DELETE FROM [Areas];
INSERT INTO [Areas] VALUES (1, '1', 'Nederland');

DELETE FROM [Projects];
INSERT INTO [Projects] VALUES (1, 'BOI', 'Riskeer calculation');

DELETE FROM [Breakwaters];
