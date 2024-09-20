## v1.0.0 (2024-09-16)

### Feat

- VRTOOL-523 Update D-Stability dependency
- VRTOOL-544 Remove 3 fields from `StandardMeasure` (#341)
- VRTOOL-545 Add flood damage to `SectionData`. (#343)
- VRTOOL-540 Check ORM version on startup (#339)
- VRTOOL-543 Remove `year` from `orm.models.Measure` (#335)
- VRTOOL-542 Added `on_delete="CASCADE"` to remaining missing models. (#332)

### Fix

- VRTOOL-590 Improve logic on database version check
- VRTOOL-578 Extend/fix probability plots
- VRTOOL-579 Include 0 and 100 as computation times (#354)
- VRTOOL-577 VRTool does not write results for all investment years (#352)

## v0.3.2 (2024-07-08)

### Fix

- [VRTOOL-561] Sections without measure are not taken for optimization (#330)
- [VRTOOL-536] Ensure that bundling no longer relies on stability & piping both being present. If not: use 0 values for current pf. Ensure dimensions are consistent. (#329)

## v0.3.1 (2024-07-03)

### Fix

- VRTOOL-522 remove creation of shsgmeasure for custom measures (#320)
- VRTOOL-521 export corrections (#323)

## v0.3.0 (2024-05-29)

### Feat

- VRTOOL-514 custom measure detail with section column (#313)
- VRTOOL-514 create custom measure per section table
- VRTOOL-495 implement safe remove of custom measures
- Added new method to remove custom measures and their results (#299)
- Added new script to add measures to a database. (#302)
- exporting CustomMeasure list of dicts does not require all T values (#297)
- Added orm controller call to add custom measures VRTOOL-346 (#287)

### Fix

- VRTOOL-518 aggregated custom measures do not have measure result (#314)
- [vrtool-510] custom measure real case (#303)

## v0.2.0 (2024-04-29)

### Feat

- Correct implement logic tr [VRTOOL-481] (#277)
- Update cost function for soil reinforcement with 0 dimensions VRTOOL-390 (#273)
- VRTOOL-344 add self retaining sheetpile logic (#266)
- VRTOOL-454 create new classes (#261)
- Vrtool 444/poc if measures dont meet cross sectional requirement (#254)
- VRTOOL-439 improve performance of greedy evaluate routine (#257)
- VRTOOL-435 separate preparation of optimization input from evaluate (#252)
- VRTOOL-406 replace optimization logic with new components (#223)
- Extended importer to include the initial assessment results for all mechanisms of the given section
- Added logic to import the measure results for optimization
- Added logic to import parameter values related to multiple computation scenarios.

### Fix

- vrtool-431 avoid failure on setting different investment years (#251)
- change path for measure import test
- fixed incorrect string for if-else.
- update wrong reference & traject  & add reference for filtered cases
- return empty list rather than None to be consistent with comment and avoid crash
- Corrected `get_failure_probability_from_scenarios` to use beta instead of `probability_of_failure` as the latter remains constant
- copy probabilities from zero measure (not year 0)
- We do a math 'isclose' comparison of maximum transition level and threshold
- Fixed logic in stability_inner_functions, adapted tests
- Corrected generation of transition level vector so that it includes the last element
- Adapted `calculate_reliability` as it was not working correctly with arrays of more than one item
- Corrected logic for creation of input object and beta calculation

## v0.1.3 (2023-12-01)

### Fix

- tweak database for stix test to include soil reinforcement in 2025
- remove incorrect worrying comment and fix handling of rare combinations of types of measures
- fixes to combination
- ensure proper ordering. Some refactoring to make steps a bit more explicit.

## v0.1.2 (2023-11-23)

### Fix

- change to removal of measures for TargetReliabilityStrategy to ensure proper handling of indices

## v0.1.1 (2023-11-10)

### Fix

- add export/import to run_all api
- take proper run_ids as the previous one was a dict that is emptied in the process.
- ensure that API works as expected. Results should also be exported
- ensure that transition_level_vector always contains at least 1 value. Raise Exception if transition is above crest.
- Removed unnecessary api / controllers call; streamlined the selection of measures when running a full model
- error in modifying attribute values.
- data handling in combination with revetments
- use itertools to combine betas rather than accessing the Series a gazillion times
- if mechanism not relevant, assume beta = 10
- improve import errors for slope parts with an invalid tan_alpha
- extend error handling for non-implemented top layer types
- extend types for asphalt
- update reference of large revetment test
- update reference for revetment test
- fix bug causing incorrect interpolations

## v0.1.0 (2023-10-23)

### Feat

- Added logic to export solutions based on a given list of measure results ids
- Added controller logic to export optimization results
- Added api file to streamline the workflows and allow for 'sandbox' exporting directly
- Added table to represent beta, time and lcc per optimization step
- Added missing table to represent the measure results per mechanism
- Added tables to represent the optimization runs and their related results
- Created exporter for solutions containing measures (SolutionsExporter)
- Created exporter for MeasureResultCollection; Renamed and created new protocols 'MeasureResultProtocol' and 'MeasureResultCollectionProtocol', so that it's easier to handle generic objects
- Added exporter for CompositeMeasures (when Measure.measures is a list of dictionaries)
- **DikeSectionReliabilityExporter**: It is now possible to export at the same time all initial assessment's reliability related to a dike section (per mechanism and general).
- **MechanismReliabilityCollectionExporter**: It is now possible to export all mechanisms reliability each into an AssessmentMechanismResult
- **SectionReliabilityExporter**: It is now possible to export section's reliability into AssessmentSectionResults

### Fix

- import was accidentally removed with merge.
- take string to select beta rather than enum object
- condition should contain not.
- "the forgotten bracket"
- add run id to ensure proper export when more than 2 runs are present in the database.
- change empty to zeros as it is not always overwritten and should be zero. Also add more explicit warning for NaN values
- add year to TakenMeasures
- add condition to distinguish measures with different years. add copy to prevent ignoring start costs.
- incorrect reference to redundant trajectdata that was not interpolated
- optimize determining min year
- reference data for revetment cases
- update for revetment in case there Revetment is not an active mechanism at the section.
- ensure revetment runs as well.
- forgot to copy a pretty essential line.
- fix error in indexing to avoid taking a 999 index.
- further changes to indexing
- fix error in indexing of measures + improvement to selection of measures to filter out measures with 0 dimensions
- fix ids in export such that correct id is used
- further wiring of selected_measure_id in optimization workflow
- export selected_measure_id rather than measure_result_id
- improve handling INVALID enum
- change df index to normalized enum name
- normalize mechanism names from imported measure reliability
- uniform method for creating list of mechanism names
- add INVALID to excluded_mechanisms in api test case
- add INVALID to excluded_mechanisms in api test case
- typo in initialization
- allow serialization of enum value
- improve building excluded_mechanisms
- resolve review comment
- improve finding enum; extend test
- improve finding enum; extend test
- resolve most review comments
- Corrected create statement
- Small correction to the measure results filter
- change mechanism string with enum names
- improve refering to enum
- imporove enum comparison
- imporove enum comparison
- improve refering to enum
- improve refering to enum
- Modified measure ORM tables and exporters to include mechanisms and reduce duplications of MeasureResultParameters
- replace input_database_path by input_database_name in config
- replace input_database_path by input_database_name in config
- replace input_database_path by input_database_name in config
- replace input_database_path by input_database_name in config
- replace input_database_path by input_database_name in config
- replace input_database_path by input_database_name in config
- replace input_database_path by input_database_name in config
- replace input_database_path by input_database_name in config
- replace input_database_path by input_database_name in config
- replace input_database_path by input_database_name in config
- replace input_database_path by input_database_name in config
- replace input_database_path by input_database_name in config
- replace input_database_path by input_database_name in config
- small change in error message

## v0.0.4 (2023-09-11)

### Feat

- Added support to import Revetment Measures
- Added revetment measure class
- Extended slope part to include specializations for grass or stone revetment
- Extended slope part to include specializations for grass or stone revetment
- Extended standard measure to include required columns for reventment measures

### Fix

- unit test. Allowed for small numerical differences (tol=1e-8)
- repair tests, references look good.
- remove entries for revetment measure at section without revetments
- update shelves for optimization tests
- reduce test runtime with less measures for revetments
- revert move of internal class. Removed revetment test from optimization tests as this is no longer supported
- modified preprocessing which should now give a complete database.
- Changed retrieval of minimum beta target; Added test to verify new behavior
- improvement in selection of measures for bundling
- improve treatment of sections with and without revetment
- ensure that bundling routines are properly accessed, changed or into and
- prevent cost values in different options dfs to be modified together
- costs for VZG and diaphragm wall.
- output of measures with revetment.
- error when cost for soil+stability screen is not a list but a float
- improve measure_combinations, logic now same as revetment_combinatiosn
- prevent filtering of measures without crest height. Ensure that both lists only have measures with the same ID
- solve deprecation warning + incorrect writing of costs
- error in filtering measures for TargetReliabilityStrategy
- improve writing of attribute values and ignore all -999. This means parameters of a measure (crest, berm etc) always have only 1 value when writing
- small change to revetmentmeasure
- update database so revetment is properly combined
- add writing section reliability to dataframe
- writing of names is now independent of the list length (could only handle 2 measures). And way less complicated
- nesting was incorrect
- ensure proper passing of different attributes of measures in revetment_combinations.
- update database such that revetment measures are combined. Also changed key.
- include revetment measures in db
- Updated _get_design_stone when calculating revetment measures.
- update shelves of TestCase1_38-1_no_housing
- update shelves for TestCase3_38-1_small
- We now 'correct' extra revetment measures without a known type
- **revetment_measure_result_builder.py**: Added missing extra measure when Overgang >= crest height.
- Fixed failing tests:
- corrected year argument:
- TeamCity does not swallow these paths, so reverted making of OptimizationSteps folder
- wrong parameter name
- height options can also have equal reliability, to ensure that measures can be shifted in time.. Added BC_list as output of overflow_bundling
- improvement of overflow_bundling
- ensure that life_cycle_cost is not modified.
- improved test for the slope part

### Refactor

- Moved standard measures to their own module

## v0.0.3 (2023-06-22)

### Feat

- Added MeasureImporter to be used by the SolutionsImporter
- Added controler to load solutions
- Added missing Table MeasureParameters
- **solutions_importer.py**: Created first solutions importer and adapted some logic from the vrtool domain
- Selected DikeTraject is now loaded from the database
- Added missing table MeasurePerSection
- **supporting_file.py**: Added new table and its own backreference
- **orm_converters.py**: Added converters to read from the database a dike traject and all its related information
- **DikeTrajectInfo**: Added orm model for dike traject info
- Added orm sql models
- **ORM**: Added peewee as orm for SQLite. Created some basic implementations to map all entities
- **vrtool_logger.py**: Added new (static) class to initialize the logger

### Fix

- update Testcase2_38-1_overflow_no_housing
- update reference for TestCase1_38-1_no_housing
- throw houses out of overflow database
- incorrect writing of measures. Small update to input db for consistency
- correct stabilityscreen parameter in StandardMeasure Table
- Set buildings index to distancefromtoe
- copy to slice warning for profile changes
- change idx to option_index
- retry, fix order of reindex
- ensure that run can be made with StabilityInner or Piping turned off.
- added safe retrieval for mechanism properties when they are not present
- added method to retrieve the years that are used for the calculation
- added mechanism collection to the section reliability
- added collection object to host all failure mechanisms and their related data
- updated logic after discussing with PO:
- updated logic after discussing with PO:
- modify order of loading the config to ensure that reuse_output can actually be used in the workflow
- remove 2 calls to SF parameter
- use correct references in plot_lcc
- force datatype in strategy_base
- data types for integers in mixed_integer_strategy
- data types for integers in solutions

## v0.0.2 (2023-03-17)

### Feat

- **vrtool_config**: We can now load and save the VrtoolConfig data from/to json files
- **run_workflows**: Created new module containing runnable workflows to assess, measure and optimize the given models
- **__main__.py**: Added endpoint for calling to the tool from CLI either locally or when installing through pip
- **vrtool_config.py**: Converted previous config.py script into a dataclass with default values. Added default unit_costs.csv file. Wrapped both files under the src/defaults module

### Fix

- added config variable to set the discount rate
- fix Sonarcloud issues
- fixed failing tests
- fixed failing test
- fixed failing test
-  fixed runtime error
- fixed errors and formatted code
- **vrtool_config**: Added __post_init__ method to allow mapping strings to paths where needed
- Small correction on using a path instead of the stem to collect years
- Corrected type hint
- **src/run_workflows**: We now create the output directory and sub directories only when needed to avoid false expectations
- **run_optimization.py**: We now safely create a directory during run_optimization
- updated reference data for failing 16-3 case
- corrected norm for DikeTraject 38-1
- added fix for the reliability calculation of piping scenarios
- **__main__.py**: Make results dir if it does not exist
- **__main__.py**: Small correction converting the model_directory arg into a path
- included fix for VRTOOL-17 to prevent an exception in the calculation of the reference test cases
- updated test as Python doesn't perform symmetric set differences
- **vrtool_config.py**: Modified parameters with list or dicts as types to be defined correctly by using field and default factories

### Refactor

- **vrtool_run_full_model.py**: Converted previous RunModel.py into RunFullModel class, added related test

## v0.0.1 (2023-02-03)
