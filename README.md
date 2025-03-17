[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![ci-install-package](https://github.com/Deltares/Veiligheidsrendement/actions/workflows/ci_installation.yml/badge.svg)](https://github.com/Deltares/Veiligheidsrendement/actions/workflows/ci_installation.yml)

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=Deltares_Veiligheidsrendement&metric=alert_status&token=483801771f090b3ceb93ef315f0332003a075970)](https://sonarcloud.io/summary/new_code?id=Deltares_Veiligheidsrendement)
![TeamCity build status](https://dpcbuild.deltares.nl/app/rest/builds/buildType:id:VrtoolSuite_CoreContinuousDelivery_RunAllTestsInParallelNoCoverageNoProfiling/statusIcon.svg)

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

## Contribution guidelines ##

To know how to collaborate within this project please refer to our [contributing page](./docs/CONTRIBUTING.md).

## Who do I talk to? ##

* Repo owner or admin
* Other community or team contact
