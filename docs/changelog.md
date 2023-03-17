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