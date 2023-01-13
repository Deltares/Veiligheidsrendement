--! Most basic input for HBN
-- KEYS TO BE FILLED: 'LocationName', HLCDID, ORIENTATION, TimeIntegration, Hmin, Hmax, Hstep
-- [Areas] ;
DELETE FROM [Areas];
INSERT INTO [Areas] VALUES (1, '1', 'Nederland');

-- [Projects] ;
DELETE FROM [Projects];
INSERT INTO [Projects] VALUES (1, 'Sprint', 'Hydra-Ring Sprint');

-- [AreaPoints] ;
DELETE FROM [AreaPoints];
--INSERT INTO [AreaPoints] ([AreaId], [RingCoordinate], [XCoordinate], [YCoordinate]) VALUES (1,   0, 123456, 654321);

-- [PresentationSections] ;
DELETE FROM [PresentationSections];
--INSERT INTO [PresentationSections] ([Name], [Description], [RingCoordinateBegin], [RingCoordinateEnd]) VALUES ('PresentationSection1', 'PresentationSection1', 12345, 54321);

-- [Sections] ;
DELETE FROM [Sections];
--INSERT INTO [Sections] ([SectionId], [PresentationId], [MainMechanismId], [Name], [Description], [RingCoordinateBegin], [RingCoordinateEnd], [XCoordinate], [YCoordinate], [StationId1], [StationId2], [Relative], [Normal], [Length]);
INSERT INTO [Sections] VALUES (  1, 1, 1, 'LocationName' , 'LocationName'  , -999 , -999, -999 , -999,  HLCDID,  HLCDID, 100, ORIENTATION  , 1); -- Non-tidal river, Rhine-dominated;																

-- [Profiles] ;
DELETE FROM [Profiles];
--INSERT INTO [Profiles] ([SectionId], [SequenceNumber], [XCoordinate], [ZCoordinate]) VALUES (1, 1, 11, 21);
INSERT INTO [Profiles] VALUES ( 1, 1,  0.000, 4.000);
INSERT INTO [Profiles] VALUES ( 1, 2, 36.000, 16.000);

-- [CalculationProfiles] ;
DELETE FROM [CalculationProfiles];
--INSERT INTO [CalculationProfiles] ([SectionId], [SequenceNumber], [XCoordinate], [ZCoordinate], [Roughness]) VALUES (1, 1, 11, 21, 0);
INSERT INTO [CalculationProfiles] VALUES (1, 1,  0.000, 4.000, 1.0);
INSERT INTO [CalculationProfiles] VALUES (1, 2, 36.000, 16.000, 1.0);

-- [Forelands] ;
DELETE FROM [Forelands];
--INSERT INTO [Forelands] ([SectionId], [SequenceNumber], [XCoordinate], [ZCoordinate]) VALUES (1, 1, -10, -3);

-- [HydraulicModels] ;
DELETE FROM [HydraulicModels];
INSERT INTO [HydraulicModels] VALUES (TimeIntegration, 1, 'WTI 2017');

-- [ProbabilityAlternatives] ;
DELETE FROM [ProbabilityAlternatives];
--INSERT INTO [ProbabilityAlternatives] ([SectionId], [MechanismId], [LayerId], [AlternativeId], [Percentage]) VALUES (1, 1, 1, 1, 100);

-- [SectionFaultTreeModels] ;
DELETE FROM [SectionFaultTreeModels];
--INSERT INTO [SectionFaultTreeModels] ([SectionId], [MechanismId], [LayerId], [AlternativeId], [FaultTreeModelId]);
-- Non-tidal river, Rhine-dominated;
INSERT INTO [SectionFaultTreeModels] VALUES (1, 1, 1, 1, 1);

-- [SectionSubMechanismModels] ;
DELETE FROM [SectionSubMechanismModels];

-- [Numerics] ; 
DELETE FROM [Numerics];
--INSERT INTO [Numerics] ([SectionId], [MechanismId], [LayerId], [AlternativeId], [SubMechanismId], [Method], [FormStartMethod], [FormNumberOfIterations], [FormRelaxationFactor], [FormEpsBeta], [FormEpsHOH], [FormEpsZFunc], [DsStartMethod], [DsIterationmethod], [DsMinNumberOfIterations], [DsMaxNumberOfIterations], [DsVarCoefficient], [NiUMin], [NiUMax], [NiNumberSteps]);
INSERT INTO [Numerics] VALUES (  1, 1, 1, 1, 1, 11, 4, 150, 0.15, 0.01, 0.01, 0.01, 2, 3, 10000, 20000, 0.1, -6, 6, 25 );

-- [DesignTables] ;
DELETE FROM [DesignTables];
--Recommended values;
-- T =   250, p = 0.004     beta = 2.65207;
-- T =   500, p = 0.002     beta = 2.87816;
-- T =  1250, p = 0.0008    beta = 3.15591;
-- T =  2000, p = 0.0005    beta = 3.29053;
-- T =  4000, p = 0.00025   beta = 3.48076;
-- T = 10000, p = 0.0001    beta = 3.71902;
--INSERT INTO [DesignTables] ([SectionId], [MechanismId], [LayerId], [AlternativeId], [Method], [VariableId], [LoadVariableId], [TableMin], [TableMax], [TableStepSize], [ValueMin], [ValueMax], [Beta]) VALUES ( 1, 1, 1, 1, 0, NULL, NULL, 10, 11, 0.01, 12, 13, 4.123 );
INSERT INTO [DesignTables]  VALUES (  1, 1, 1, 1, 3, 26, NULL, Hmin, Hmax, Hstep, 2.0, 4.0, 3.71902 );

											
-- [VariableDatas] ;
DELETE FROM [VariableDatas] ;
INSERT INTO [VariableDatas] VALUES (1, 1, 1, 1, 26, 0, 0, 0, NULL, NULL, NULL, 1, 0, 300);

DELETE FROM [SetUpHeights];
DELETE FROM [CalcWindDirections];
DELETE FROM [Swells];
DELETE FROM [WaveReductions];
DELETE FROM [Fetches];
DELETE FROM [ForelandModels];

