[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3129/)
[![ci-install-package](https://github.com/Deltares/Veiligheidsrendement/actions/workflows/ci_installation.yml/badge.svg)](https://github.com/Deltares/Veiligheidsrendement/actions/workflows/ci_installation.yml)

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=Deltares_Veiligheidsrendement&metric=alert_status&token=483801771f090b3ceb93ef315f0332003a075970)](https://sonarcloud.io/summary/new_code?id=Deltares_Veiligheidsrendement)
![TeamCity build status](https://dpcbuild.deltares.nl/app/rest/builds/buildType:id:VrtoolSuite_CoreContinuousDelivery_RunAllTests/statusIcon.svg)

# Veiligheidsrendement #

This is the repository as developed in the AllRisk programme to apply the veiligheidsrendementmethode for optimal planning of flood defence systems.

## What is this repository for?

* Quick summary
* Version

## How do I get set up? ##

__Important!__ The following installation steps are written based on a Windows environment. When using other systems (which should be possible) it might be required to use different commands. However, the fundamental of the installation steps should remain the same. This meaning, no additional packages or libraries should be required. If problems would arose during your installation, please contact the maintainers of the tool.

### Sandbox / Endpoint

When you only require the `VeiligheidsrendementTool` package to be used as a whole, and not for developments, we advise to directly use the latest greatest release, or directly the latest available version from `main` as follows:

1. Latest available `main`:
```bash
pip install git+https://github.com/Deltares/Veiligheidsrendement.git
```

2. Specific `Veiligheidsrendement` version, add `@version-tag` to the previous command, for instance install tag `v0.0.1` (__Proof of Concept__ previous to this GIT repository):
```bash
pip install git+https://github.com/Deltares/Veiligheidsrendement.git@v0.0.1
```
| You can also do the above with a commit-hash for development branches (e.g.:`@40bd07d`)


### Development mode

We recommend you to check our `CONTRIBUTING.md` document and its [installation steps section](./docs/CONTRIBUTING.md#install-before-contributing).


### Dependencies / Pre-requirements.

#### D-Stability
It is the responsibility of the user to have their own DStabilityConsole binaries locally available in order to run it with the `vrtool`.
We are using the 2022.01 release of D-Stability.
For a correct functioning we advise you to have a look on our tutorial section
[Running a D-Stability model](https://deltares-research.github.io/VrtoolDocumentation/Achtergronden/Betrouwbaarheidsmodellen/Binnenwaartse%20macrostabiliteit.html#d-stability).

#### openturns
We found out a hard dependency when working under a Windows environment with the [library `openturns`](https://openturns.github.io/www/index.html), which forced us to work under the version 1.19. This is automatically resolved for you when following the steps specified for [development mode](#development-mode).
When using your own environment, you might have to follow the openturns installation steps for version 1.19.

### How to run tests
Tests can be run with the pytest command `pytest run`. However, when working under a [development mode](#development-mode) environment, we advise to run the command `poetry run pytest` instead.


## Endpoint usage
 
When using `Veiligheidsrendement` as a package (`vrtool`) you can run it directly from the command line as follows:

```cli
python -m vrtool {desired_run} {CONFIG_FILE}
```
The run options are:
- `assessment`: Runs a validation with the database and settings specified in the `CONFIG_FILE`.
- `measures`: Calculates the effect of the provided measures on all specified mechanisms in the model.
- `optimization`: Runs an optimization of the model based on the previous measures run.
- `run_full`: Runs all the steps above sequentially.

The arguments are:
- `CONFIG_FILE` (required): Absolute path to the `*.json` file containing the configuration to be run.

It is also possible to check all the above possibilities via the `--help` argument in the command line:
```cli
python -m vrtool --help
```

Or the running version of vrtool:
```cli
python -m vrtool --version
```

## Docker usage

We have a docker container available in our Deltares Harbor which also includes the DStability console. Its commands are exactly the same as for the [endpoint usage](#endpoint-usage) but we need to provide the docker (podman) commands as well as to mount the model to run. Some examples here:

```cli
>podman run -it vrtool --help
Usage: python -m vrtool [OPTIONS] COMMAND [ARGS]...

  Set of general available calls for VeiligheidsrendementTool.

Options:
  --version         Show the version and exit.
  --externals PATH  Path to externals directory.
  --help            Show this message and exit.

Commands:
  assessment      Assesses the model with the given configuration file.
  measures        Calculates the reliability and cost for all measures...
  migrate_db      Migrate the provided database file.
  migrate_db_dir  Migrates all provided database files in a directory.
  optimization    Optimizes the model measures with the given...
  run_full        Full run of the model with the given configuration.
```

By default our container will run using the provided 'externals' directory ( found in `/app/externals`), but you can always overriding it with `--externals {your_mounted_externals_location}`

We will execute a `run_full` command, for this purpose we will create a directory `docker_case` with the database from our `38-1_two_river_sections` saved as `vrtool_input.db` and a `docker_config.json` file containing the following:

```json
{
    "input_directory": ".",
    "input_database_name": "vrtool_input.db",
    "traject": "38-1",
    "output_directory": "./output",
    "excluded_mechanisms": [
        "REVETMENT",
        "HYDRAULIC_STRUCTURES"
    ]
}
```

We can now run the container by mounting our directory ( `docker_case` ) to `/model`, then we need to provide the mounted path where the configuration is located `/model/docker_config.json` as:
```cli
>podman run -v docker_case:/model -it vrtool run_full /model/docker_config.json
2025-12-09 01:56:08 PM - [__main__.py:38] - root - INFO - Start logging vanuit /app/vrtool_logging_20251209_1356.log
2025-12-09 01:56:08 PM - [__main__.py:132] - root - INFO - Start volledige berekening met configuratie /model/docker_config.json!
2025-12-09 01:56:08 PM - [__main__.py:41] - root - INFO - Config externals path: None.
2025-12-09 01:56:08 PM - [__main__.py:42] - root - INFO - CLI externals path: /app/externals.
2025-12-09 01:56:08 PM - [api.py:259] - root - INFO - Start beoordeling & doorrekenen maatregelen.
2025-12-09 01:56:09 PM - [orm_controllers.py:214] - root - INFO - Bestaande beoordelingsresultaten verwijderd.
2025-12-09 01:56:09 PM - [run_safety_assessment.py:30] - root - INFO - Start stap 1: beoordeling & projectie veiligheid
2025-12-09 01:56:09 PM - [run_safety_assessment.py:59] - root - INFO - Stap 1 afgerond.
2025-12-09 01:56:09 PM - [orm_controllers.py:196] - root - INFO - Resultaten beoordeling & projectie geexporteerd naar database.
2025-12-09 01:56:09 PM - [orm_controllers.py:243] - root - INFO - Bestaande resultaten voor maatregelen verwijderd.
2025-12-09 01:56:09 PM - [orm_controllers.py:260] - root - INFO - Bestaande optimalisatieresultaten verwijderd.
2025-12-09 01:56:09 PM - [run_measures.py:44] - root - INFO - Start stap 2: bepaling effecten en kosten van maatregelen.
...
2025-12-09 01:57:17 PM - [run_optimization.py:115] - root - INFO - Start stap 3: Bepaling maatregelen op trajectniveau.
2025-12-09 01:57:17 PM - [run_optimization.py:63] - root - INFO - Start optimalisatie van maatregelen voor Veiligheidsrendement.
2025-12-09 01:57:18 PM - [greedy_strategy.py:679] - root - INFO - Enkele maatregel in optimalisatiestap 0 (BC-ratio = 5580.15)
2025-12-09 01:57:18 PM - [greedy_strategy.py:679] - root - INFO - Enkele maatregel in optimalisatiestap 1 (BC-ratio = 581.94)
2025-12-09 01:57:18 PM - [greedy_strategy.py:679] - root - INFO - Enkele maatregel in optimalisatiestap 2 (BC-ratio = 119.64)
2025-12-09 01:57:18 PM - [greedy_strategy.py:723] - root - INFO - Gebundelde maatregelen voor overslag in optimalisatiestap 3 (BC-ratio = 37.15)
2025-12-09 01:57:18 PM - [greedy_strategy.py:723] - root - INFO - Gebundelde maatregelen voor overslag in optimalisatiestap 4 (BC-ratio = 11.28)
2025-12-09 01:57:18 PM - [greedy_strategy.py:723] - root - INFO - Gebundelde maatregelen voor overslag in optimalisatiestap 5 (BC-ratio = 5.85)
2025-12-09 01:57:18 PM - [greedy_strategy.py:679] - root - INFO - Enkele maatregel in optimalisatiestap 6 (BC-ratio = 0.42)
2025-12-09 01:57:18 PM - [greedy_strategy.py:723] - root - INFO - Gebundelde maatregelen voor overslag in optimalisatiestap 7 (BC-ratio = 0.25)
2025-12-09 01:57:18 PM - [greedy_strategy.py:679] - root - INFO - Enkele maatregel in optimalisatiestap 8 (BC-ratio = 0.78)
2025-12-09 01:57:18 PM - [greedy_strategy.py:679] - root - INFO - Enkele maatregel in optimalisatiestap 9 (BC-ratio = 0.18)
2025-12-09 01:57:18 PM - [greedy_strategy.py:776] - root - INFO - Totale rekentijd voor veiligheidsrendementoptimalisatie 0.92 seconden
2025-12-09 01:57:18 PM - [run_optimization.py:82] - root - INFO - Start bepaling referentiemaatregelen op basis van Doorsnede-eisen.
2025-12-09 01:57:20 PM - [run_optimization.py:133] - root - INFO - Stap 3: Bepaling maatregelen op trajectniveau afgerond
2025-12-09 01:57:21 PM - [orm_controllers.py:464] - root - INFO - Resultaten geexporteerd.
2025-12-09 01:57:21 PM - [api.py:272] - root - INFO - Berekening afgerond.
```

The results of our calculation can be found in our __local__ directory `docker_case/output` and in the provided database `docker_case/vrtool_input.db`.


## Contribution guidelines ##

To know how to collaborate within this project please refer to our [contributing page](./docs/CONTRIBUTING.md).

## Who do I talk to? ##

* Repo owner or admin
* Other community or team contact
