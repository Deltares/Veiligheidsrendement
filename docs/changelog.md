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