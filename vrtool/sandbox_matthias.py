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
_vrtool_dir = Path(r"C:\Users\hauth\OneDrive - Stichting Deltares\Desktop\projects\VRTools\TestCases")
_input_model = _vrtool_dir / "TestCase1_38-1_no_housing"
# _input_model = _vrtool_dir / "integrated_SAFE_16-3_small_FULL"
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
_selected_traject = DikeTraject.from_vr_config(_vr_config)
assert isinstance(_selected_traject, DikeTraject)


dike_section = _selected_traject.sections[0]
print('mechanism data 1st cs:', dike_section.mechanism_data)


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
#
# # Step 3. Optimization.
# _optimization = RunOptimization(_measures_result, plot_mode=_plot_mode)
# _optimization_result = _optimization.run()
# assert isinstance(_optimization_result, ResultsOptimization)