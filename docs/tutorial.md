# Veiligheidsrendement tutorial
This file works as a tutorial to demonstrate how to import and use the `Veiligheidsrendement Tool` (vrtool) package from an external script. Therefore, without having to checkout the whole repository from [Github](https://github.com/Deltares/Veiligheidsrendement/)

## Setup
We will create our custom `Anaconda` environment file `environment.yml`:
```yml
name: vrtool_env
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.10
  - pip
  - conda-forge::openturns=1.19
```
Which we can create in our working directory as follows: `conda env create -f environment.yml -p vrtool_env`.

After the environment is created we need to activate it: `conda activate vrtool_env`.

## Installing the package
We proceed now by installing the `vrtool` package, for this example we will install the version corresponding to the latest tag `v0.0.2`:

```
pip install git+https://github.com/Deltares/Veiligheidsrendement.git@v0.0.2
```
## Model preparation
For the next sections we will assume that the dataset for a dike traject is present in our working directory, in particular we will be using the `integrated_SAFE_16-3_small` model present as test data in the [vrtool test bench](https://github.com/Deltares/Veiligheidsrendement/tree/main/tests/test_data/integrated_SAFE_16-3_small)

Since `vrtool` version `0.0.2` it is required to have a `.json` configuration file defining concrete properties, in particular we **need** to define which traject will be selected. For example, our model example contains its configuration in `custom_config.json`:
```json
{
    "traject": "16-3",
    "externals": "C:\\repos\\external_libraries",
}
```
> At the moment the software **assumes** only one `json` file is present in the root model directory.

## Running a D-Stability model.
For a `D-Stability` run it is needed to add its respective console as done for [GeoLib](https://deltares.github.io/GEOLib/latest/user/setup.html) (`vrtool` depends on `d-geolib`). The console directory needs then to be located directly under our `externals` directory. GeoLib implicitly adds to our externals path the subdirectory `\\DStabilityConsole` and the console `D-Stability Console.exe`.

This means, that in our directory `"C:\\repos\\external_libraries` we will need to place the `DStabilityConsole` and all of its contents such as the path to the console can be found`C:\\repos\\external_libraries\\DStabilityConsole\\D-Stability Console.exe`. We do not need to provide this path, as its __implicit__ by GeoLib.

## Running the CLI
If the installation was correct we should now be able to run the `vrtool` module directly from the command line. Let's quickly verify it with the help command:

```console
(vrtool_env) D:\repos\vrtool-example>python -m vrtool --help
Usage: python -m vrtool [OPTIONS] COMMAND [ARGS]...

  Set of general available calls for VeiligheidsrendementTool.

Options:
  --help  Show this message and exit.

Commands:
  assessment    Assesses the model in the given directory.
  measures      Calculates all measures for all specified mechanisms in...
  optimization  Optimizes the model measures in the given directory.
  run_full      Full run of the model in the given directory.

(vrtool_env) D:\repos\vrtool-example>python -m vrtool --help
Usage: python -m vrtool [OPTIONS] COMMAND [ARGS]...

  Set of general available calls for VeiligheidsrendementTool.

Options:
  --help  Show this message and exit.

Commands:
  assessment    Assesses the model in the given directory.
  measures      Calculates all measures for all specified mechanisms in...
  optimization  Optimizes the model measures in the given directory.
  run_full      Full run of the model in the given directory.
```

It works, which means we can do a full run of a model we have in our repository such as `vrtool\integrated_SAFE_16-3_small`, which corresponds to a dataset related to traject `16-3`.

```console
(vrtool_env) D:\repos\vrtool-example>python -m vrtool run_full integrated_SAFE_16-3_small
Single measure in step 0
Single measure in step 1
...
Elapsed time for greedy algorithm: 4.34426474571228
```

## "Sandboxing"
This way of working gives the modeler the option to fully control how to run a `VeiligheidsRendement` model. Let's demonstrate it by simply replicating one of the tests present in the `vrtool` repository:

```python
from pathlib import Path
from shutil import rmtree
from vrtool.defaults.vrtool_config import VrtoolConfig
from vrtool.run_workflows.vrtool_plot_mode import VrToolPlotMode
from vrtool.flood_defence_system.dike_traject import DikeTraject
from vrtool.run_workflows.measures_workflow.results_measures import ResultsMeasures
from vrtool.run_workflows.measures_workflow.run_measures import RunMeasures
from vrtool.run_workflows.optimization_workflow.results_optimization import (
    ResultsOptimization,
)
from vrtool.run_workflows.optimization_workflow.run_optimization import RunOptimization
from vrtool.run_workflows.safety_workflow.results_safety_assessment import (
    ResultsSafetyAssessment,
)
from vrtool.run_workflows.safety_workflow.run_safety_assessment import (
    RunSafetyAssessment,
)

# 1. Define input and output directories..
_vrtool_dir = Path(__file__).parent
_input_model = _vrtool_dir / "integrated_SAFE_16-3_small"
assert _input_model.exists(), "No input model found at {}".format(_input_model)

_results_dir = _vrtool_dir / "sandbox_results"
if _results_dir.exists():
    rmtree(_results_dir)

# 2. Define the configuration to use.
_vr_config = VrtoolConfig()
_vr_config.input_directory = _input_model
_vr_config.output_directory = _results_dir
_vr_config.traject = "16-3"
_plot_mode = VrToolPlotMode.STANDARD

# 3. "Run" the model.
# Step 0. Load Traject
_selected_traject = DikeTraject(_vr_config)
assert isinstance(_selected_traject, DikeTraject)

# Step 1. Safety assessment.
_safety_assessment = RunSafetyAssessment(
    _vr_config, _selected_traject, plot_mode=_plot_mode
)
_safety_result = _safety_assessment.run()
assert isinstance(_safety_result, ResultsSafetyAssessment)

# Step 2. Measures.
_measures = RunMeasures(_vr_config, _selected_traject, plot_mode=_plot_mode)
_measures_result = _measures.run()
assert isinstance(_measures_result, ResultsMeasures)

# Step 3. Optimization.
_optimization = RunOptimization(_measures_result, plot_mode=_plot_mode)
_optimization_result = _optimization.run()
assert isinstance(_optimization_result, ResultsOptimization)
```