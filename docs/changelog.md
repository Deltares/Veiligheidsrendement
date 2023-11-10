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

- add run id to ensure proper export when more than 2 runs are present in the database.
- change empty to zeros as it is not always overwritten and should be zero. Also add more explicit warning for NaN values
- add year to TakenMeasures
- add condition to distinguish measures with different years. add copy to prevent ignoring start costs.
- incorrect reference to redundant trajectdata that was not interpolated
- optimize determining min year
- update for revetment in case there Revetment is not an active mechanism at the section.
- ensure revetment runs as well.
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
- typo in initialization
- allow serialization of enum value
- improve building excluded_mechanisms
- improve finding enum;
- resolve most review comments
- Corrected create statement
- Small correction to the measure results filter
- change mechanism string with enum names
- improve refering to enum
- imporove enum comparison
- improve refering to enum
- Modified measure ORM tables and exporters to include mechanisms and reduce duplications of MeasureResultParameters
- replace input_database_path by input_database_name in config

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

### Fix

- added config variable to set the discount rate
- fix Sonarcloud issues
- fixed failing tests
- **vrtool_config**: Added __post_init__ method to allow mapping strings to paths where needed
- Small correction on using a path instead of the stem to collect years
- fixed failing test
- fixed failing test
- Corrected type hint
-  fixed runtime error
- fixed errors and formatted code
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

### Feat

- **vrtool_config**: We can now load and save the VrtoolConfig data from/to json files
- **run_workflows**: Created new module containing runnable workflows to assess, measure and optimize the given models
- **__main__.py**: Added endpoint for calling to the tool from CLI either locally or when installing through pip
- **vrtool_config.py**: Converted previous config.py script into a dataclass with default values. Added default unit_costs.csv file. Wrapped both files under the src/defaults module

### Refactor

- **vrtool_run_full_model.py**: Converted previous RunModel.py into RunFullModel class, added related test

## v0.0.1 (2023-02-03)
- Initial tag based on previous version from Wouter Jan Klerk (wouterjan.klerk@deltares.nl)