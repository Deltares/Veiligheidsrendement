# Contribution guidelines

We thank you already for your interest and time to contribute within this project.

The first step for contributing to [vrtool](https://github.com/Deltares/Veiligheidsrendement) is to [install it for development](#install-before-contributing). Then you can proceed to [create a JIRA issue](#how-to-create-an-issue) to describe (and possible discuss) your contribution(s) followed-up by creating a [related GitHub branch](#creating-new-branches). Changes can then be [committed](#commiting-changes) following the vrtool's [code standards](#code-standards) and don't forget that [before making a pull request](#before-making-a-pull-request) you should consider to [add documentation](#adding-documentation).

## Install before contributing
If you have not installed the `vrtool` for development please do so now: 

1. Checkout the code from github in a directory of your choice. You can either do this by downloading the source zip or (better) using git, for instance:
    ```bash
    cd C:\repos
    git clone https://github.com/Deltares/Veiligheidsrendement.git vrtool_repo
    ```
    | Note, the above steps are based on a Windows setup. If you are not familiar with Git, we recommend using the [GitHub desktop tool](https://desktop.github.com/).

2. Navigate to your `Veiligheidsrendement` local directory and then install the `vrtool` package with [Pixi](https://pixi.sh/latest/installation/) (or check [other options](#other-installation-options)), we will install the `dev` environment that guarantees we have all the test packages:    
    ```bash
    cd C:\repos\vrtool_repo
    pixi install -e dev
    ```
### Other installation options.

### Using the Docker dev container.
With each succesful commit to `main` we publish a new development docker container that can be used as a means to speed up the installation steps in a controlled environment. This docker development container includes the "test externals" (at the moment of writing this is only the `DStability` **linux** kernel) and a conda environment with poetry already installed in it.

The container is accessible to all deltares contributors at the [Deltares Harbor](containers.deltares.nl/gfs-dev/vrtool_dev:latest)

To download and use it, assuming a running [Docker desktop](https://www.docker.com/products/docker-desktop/) instance, you may follow the following steps:

1. Pull and rename the docker dev image from `containers.deltares.nl/gfs-dev/vrtool_dev:latest`:
    ```bash
    docker pull containers.deltares.nl/gfs-dev/vrtool_dev:latest
    docker image tag containers.deltares.nl/gfs-dev/vrtool_dev:latest vrtool_dev
    ```
    > You may change `:latest` with any other development tag available. This is useful during development of features with new dependencies.

2. Access the docker and mount your current checkout.
    ```bash
    docker run -v .:/usr/src/app/ -v docker_env:/usr/src/.env -it vrtool_dev
    ```
    > -v Will mount the left side (yor local disk) into the right one (docker virtual volume). Mounting the environment allows you to reuse it at anytime.

    > -it Will give you access to the docker console.

3. Copy the "test externals" into the test directory (so tests can be run):
    ```bash
    cp -r /usr/src/test_externals /usr/src/app/tests/test_externals
    ```

3. Install your checkout.
    ```bash
    pixi install -2 dev
    ```

## Before making a pull request.
1. Make sure you followed our [code standards](#code-standards).
2. Make sure you [added tests](#adding-tests) to cover the added / updated functionality.
3. Make sure the Quality Gate of [SonarCloud](https://sonarcloud.io/project/overview?id=Deltares_Veiligheidsrendement) is succesful.
    1. no new bugs or code smells are introduced.
    2. code coverage has not dropped (and hopefully has increased).
4. Make sure there are no failing tests in [TeamCity](https://dpcbuild.deltares.nl/project/VrtoolSuite_CoreCiCd?branch=%3Cdefault%3E&buildTypeTab=overview&mode=builds).
5. Make sure you [added documentation](#adding-documentation).
6. [Create a pull request](#creating-a-pull-request)

## Reporting a problem

We make use of a [GitHub Issues](https://github.com/Deltares/Veiligheidsrendement/issues) to report new features, tasks, general improvements and of course bugs.

### How to create an issue?

An issue can be easily be created from JIRA, in case this is not possible please communicate it to the project administrator(s). For now, we categorize the issues based on:

* __Bug__: an error or malfunctioning in the code is found and needs a fix.
* __New feature__: a new development is needed to fulfill a (non) functional requirement. This usually consists of writing entirely new code.
* __Improvement__: same as new feature although in this case consists of partially re-writing the existing code to enhance its performance and functionalities.
* __Task__: usually related to non-development related actions, such as setting up Continuous Integration platforms or consulting over the project.
* __Documentation__: anything related to writing or updating the project's documentation.

When creating an issue please try to describe it as best as possible including:
* Which version of the  `vrtool` is affected by it.
* Which version of the `vrtool`is expected to fulfill it.
* What is the current situation. What is happening?
* What is the desired situation. What needs to be done?
* When is the issue considered as completed (completion criteria).

In addition to the above points:
* For code developments (bug / new / improvement) please also add test data whenever possible.
* For bugs or improvements, describe the exact steps (and system used) needed to reproduce the current situation.

### How to manage the board? 

The management of the board is (mostly) restricted to the Scrum Master and the Product Owner. The latter will prioritize the issues present in the backlog and prepare them prior to a development sprint.

A contributor is expected to be working at most on one issue at the time, however they can also participate by means of reviewing or testing other contributor's work.

Only the issues present in a sprint should be picked up for development. An issue's workflow usually is `Development -> Review -> Test -> Done`. Review and test will be carried out by different contributors, potentially asking the original contributor to add changes.


## Use of GitHub

At the moment of writing this document the [vrtool repository](https://github.com/Deltares/Veiligheidsrendement) is private and its access constrained to those in the official development team. If you wish to contribute, please contact the repository and project administrator(s) first.

### Creating new branches

New branches are encouraged based on the _dare to share_  principle, this allows the rest of contributors to be aware of your changes as soon as possible. However, please do not forget to create a related issue as described in the [above section](#when-should-issues-be-created).

All created branches should adhere to the pattern `prefix\##_title_in_snake_case`. Where:

* `prefix`, is a descriptive string representing the type of issue. The current accepted prefixes are the following:
    - __main__: This prefix is not to be used as containts the latest (greatest) changes. To contribute into this branch you will have to do a pull-request from your own branch.
    - __feature/*__: for issues tagged as "new feature" or "improvement".
    - __bug/*__:  for issues tagged as "bug".
    - __poc/*__: for research and development issues outside an official sprint.
    - __chore/*__: for issues tagged as "task" or "improvement" that are trivial such as cleaning up code or modifying a few tests. 
    - __ci/*__: for issues tagged as "task", "improvement" or "new feature" related to CI/CD pipelines.
    - __docs/*__: for issues tagged as "documentation".
* `##`, is the issue number that can be found in its own page (for instance https://github.com/Deltares/Veiligheidsrendement/issues/417) would be `471`.
* and `title_in_snake_case` is a short abstract of the title (they can be often too long) in the snake case format, issue `71 Create collaboration How to.` should be `_create_collaboration_how_to`.

So, as an example the documentation issue `71 Create collaboration How to` would be carried out in a branch named as `docs/71_create_collaboration_how_to`.

### Adding dependencies

This project makes use of [Pixi](https://pixi.sh). In `vrtool`, we distinguish these types of dependencies.
- general ( `pixi add _dependency_`) . A dependency that __needs__ to be distributed with the package because it is required for a correct functioning of the tool. Think of packages such as `pandas` or `numpy`.
- test (`pixi add _dependency_to_add_ --feature test`). A dependency only used during testing, think of `pytest` or libraries used to generate code coverage reports.
- dev (`pixi add _dependency_to_add_ --feature dev` ). Think of dependencies needed by developers to help in their tasks, such as `black`, `isort`. 

Each dependency group gets installed as a separte environment:
- `pixi install`: installs the default "feature" dependencies for the internally defined python version.
- `pixi install -e test`:  installs the default and test "features".
- `pixi install -e dev`: installs the default, test and dev "features", this is the want you want for contributing.

### Commiting changes

In this repository we try to commit following the [Conventional Commits standard](https://www.conventionalcommits.org/en/v1.0.0/). This allows us for easy interpretation of ones contribution as well as to automatically maintain our `CHANGELOG.md`.

In short, we encourage the contributor to add a prefix to their commits such as `prefix: description of the commit`. The prefixes we currently use are as follow:

- `feat`. For feature changes (new functionalities).
- `fix`. For changes related to fixing a bug.
- `chore`. For changes related to trivial tasks like code formatting or changing method names.
- `test`. For changes related to creation or update of tests.
- `docs`. For changes related to creation or update of documentation.
- `ci`. For changes related to the continuous integration builds, think of GitHub workflows or the project's `.toml` file.

### Creating a pull-request

Pull requests will be used to do merges back to `main`. In a pull request please describe as best as possible what has been accomplished.

We make use of GitHub pull requests for code review. Each pull-request is synchronized with both [TeamCity](https://dpcbuild.deltares.nl/project/Vrtool?branch=%3Cdefault%3E&buildTypeTab=overview&mode=builds) and [SonarCloud](https://sonarcloud.io/project/overview?id=Deltares_Veiligheidsrendement) to ensure its Continuous Integration status.

Do not forget to notify the other contributors about the Pull-Request by moving its related [Jira issue](#jira-board) to the "Review" column.

### Approving and merging a pull-request.

Only after a succesful review and "green" quality gates a pull-request will be merged back to `main`. This approval will be explicetly set by the code reviewer and it can directly merge the changes or wait for the original contributor to do so.

When "merging a pull-request" three options are displayed. Our preferred way will be, unless otherwise stated, to "Create a merge commit".

After merging a pull-request its related source branch will be deleted from the repository and a [Github workflow](../.github/workflows/normalize_code.yml)) will be run.

### Creating a release.

On shipment of the software a formal release is created.

1. Create (and afterwards polish) the `changelog.md`:
    ```bash
    cz changelog --unreleased-version="v<x>.<y>.<z>"
    ```
    where `x`, `y`, `z` are the major, minor, patch version (e.g. v1.0.3).

2. Update the version and create a tag:
    ```bash
    cz bump --increment <increment_type>
    ```
    where `increment_type` is `MAJOR`, `MINOR` or `PATCH`, depending on the impact of the release.
    Note the resulting version should be identical to the one given in the previous step.

3. Push the release:
    ```bash
    git push tags
    git push --force
    ```

## Code standards

In general, we adhere to the [Zen of Python](https://peps.python.org/pep-0020/#id3) and we use the [Google convention](https://google.github.io/styleguide/pyguide.html) as a base for our coding standards. Those points where we differ from the _Google convention_ are documented below. We consider this document to be a living document, so it is subject to discussion and potential changes.

When we talk about normalization we refer to standardizing how we name, describe, reference and use the following <span ID="items-list">items</span>:
- a package (folder),
- a module (file),
- a class,
- a method,
- a parameter,
- a property,
- a variable,


Code formatting happens in its majority with a [Github workflow](../.github/workflows/normalize_code.yml)  which is enforced after each succesful [pull-request merge](#approving-and-merging-a-pull-request) to `main`. This can be at any time locally done running the line: `poetry run isort . && poetry run black .`.

Our own agreements for `vrtool` code standards are as follows and will be looked up during a pull-request review:

### Naming conventions

In general we use the following standards:
- [PascalCase](https://en.wiktionary.org/wiki/Pascal_case#English), for class names.
- [snake_case](https://en.wikipedia.org/wiki/Snake_case), for the rest.

Although in Python 'private' and 'public' is a vague definition, we often use the underscore symbol `_` to refer to objects that are not meant to be used outside the context where they were defined. For instance:

- We underscore method's names when they are not meant to be used outisde their container class.
- In addition, we suggest to underscore the variables defined within a method to (visually) differenciate them from the input arguments (parameters):
    ```python
    def example_method(param_a: float, param_b: float) -> float:
        _sumat = param_a + param_b
        return _sumat
    ```

### Module (file) content

In general:

- One file consists of one (and only one) class.
- The file containing a class will have the same name (snake case for the file, upper camel case for the class).

Some exceptions:

- An auxiliar dataclass might be eventually defined in the same file as the only class using (and referencing) it.
- Test classes may contain mock classes when they are only to be used within said test-file.

### Describing an [item](#items-list)

- Packages can be further described with `README.md` files.
- Modules are described with docstrings using the [google docstring convention](https://gist.github.com/redlotus/3bc387c2591e3e908c9b63b97b11d24e)
- We prefer explicit over implicit declaration.
    - Use of [type hinting](https://docs.python.org/3/library/typing.html)
- Classes are __always__ described with docstrings, its properties also have descriptive names and have explicit types using [type hints](https://docs.python.org/3/library/typing.html).
- Methods contain a clear descriptive name, its arguments (parameters) contain [type hints](https://docs.python.org/3/library/typing.html) and in case it is a 'public' method its signature has a description following the [google docstrings](https://google.github.io/styleguide/pyguide.html) formatting.

### Protocols

We use [protocols](https://docs.python.org/3/library/typing.html#typing.Protocol) to describe the behavior of classes and enable polymorphism.

### Do's and dont's


#### Built-in functions:

We use built-in functions when they help us achieve more efficient code, with the condition the code remains readable.

In case a complex built-in function is used it is strongly advised to add a comment explaining what this function is doing.

Using built-in functions should still allow the developer to easily debug the wrapping method.


#### Dataclasses

We define a [dataclass](https://docs.python.org/3/library/dataclasses.html) when we require a repeating data structure that contains multiple properties potentially with default values. We consider a dataclass responsible only for exposing its own context, therefore not for modifying its own state or the one from other objects.

- Do:
```python
from dataclasses import dataclass

@dataclass
class MyExampleDataclass:
    page_width: float
    page_height: float
    page_margin: float = 0.2

    @property
    def left_margin(self) -> float:
        _margin_size = self.page_margin * self.page_width
        return self.page_width - (_margin_size / 2)
```

- Don't:
```python
from dataclasses import dataclass

@dataclass
class MyExampleDataclass:
    page_width: float
    page_height: float
    left_margin: float = 0.0
    page_margin: float = 0.2

    def set_margin(self) -> None:
        _margin_size = self.page_margin * self.page_width
        self.left_margin = self.page_width - (_margin_size / 2)
    
    @staticmethod
    def set_page_left_margin(page: MyExampleDataclass, page_margin: float):
        page.page_margin = page_margin
        page.set_margin
```

#### Class methods

Class methods can be used to replace the multiple `__init__` needs that are often present in other languages like `C#`.

It is suggested to adhere to the method naming convention to have a clear understanding on how the object is to be created.

```python

class MyExample:

    def __init__(self):
        ...
    
    @classmethod
    def from_pandas(cls, pandas_df: pd.DataFrame) -> MyExample:
        _my_example = cls()
        ...
        return _my_example

```

#### Inner functions

An inner function, or a method within a method, can be helpful to reduce code duplicity within a method whilst reusing the variables defined within the parent method's context. When an inner function does not make use of anything from the context it might better be declared as a 'sibling' static method.

- Example:
```python
def example_method(param_a: float, param_b: int) -> float:
    return sum([v * param_a for v in range(0, param_b)])
```

- Do:
```python
def example_method(param_a: float, param_b: int) -> float:
    def multiply(value: float):
        return value * param_a
    return sum([multiply(v) for v in range(0, param_b)])
```
- Don't:
```python
def example_method(param_a: float, param_b: int) -> float:
    def multiply(value: float, param_value: float):
        return value * param_value
    return sum([multiply(v, param_a) for v in range(0, param_b)])
```

#### Using flags

Using flags in a method is discouraged (yet not forbidden), think on creating two different methods for each alternative and having an `if-else` at the caller's level instead.

When the parameter (most times `bool`) is used to determine the workflow of a method then is better not to go for it.

- Do:
```python
def _get_range(from_value: float, to_value: float) -> list[float]:
    return range(from_value, to_value)
def example_method(param_a: float, param_b: float) -> list[float]:
    return _get_range(param_a, param_b)
def example_method_reversed(param_a: float, param_b: float) -> list[float]:
    return _get_range(param_b, param_a)
x = 4.2
y = 2.4
_generated_range = example_method_reversed(x, y) if x > y else example_method(x, y)
```
- Better do not for new functionalities:
```python
def example_method(param_a: float, param_b: float, is_reversed: bool) -> list[float]:
    if is_reversed:
        return range(param_b, param_a)
    return range(param_a, param_b)
x = 4.2
y = 2.4
_generated_range = example_method(x, y, x > y)
```

#### Nested loops and if-elses

> Flat is better than nested.</br>
> Sparse is better than dense.</br>
> Readability counts.</br>
> ["Zen of Python"](https://peps.python.org/pep-0020/#id3) 

Keep nested `for-loops` and `if-else` statements as flat as possible. In order to reduce complexity we encourage extracting, whenever possible said `for-loops` and `if-else` logic into other methods so to improve their readability.

- In some cases better algorithmic approaches can improve readability, think of:
    - Inversion to reduce nesting on `if-elses`
    - Pre-initialization of variables.
    - Filtering of collections prior to a loop.


## Adding tests

Contributors should also extend the test bench for the `vrtool`. Vrtool usages [PyTest](https://docs.pytest.org/en/7.3.x/) to discover and run tests. Creating tests should be done following the available pytest formats.

The [tests module](../tests/) "mirrors" the directory structure of the `vrtool` one. Hence, for each new module a new "test module" (with same name) should be created, for a new source file a new test file should be created (with the `test_` prefix) and each class should be tested with a related test class (with the `Test` prefix such as `class TestMyClass:`).

When required to add test data for referencing or as input, you may do so by adding it to the [test_data](../tests/test_data/) directory and then reference to it via the tests root import `from tests import test_data`. Similarly, when generating new files of any type during a test, please try to output it directly to the `tests_results` directory, which can also be imported as `from tests import test_results`.

We only distinguish, at the moment, two different categories of tests in `vrtool`. Regular tests (untagged) and `slow` tests. You can tag a `slow` test when its run takes more than one minute such as:

```python
@pytest.mark.slow
def given_real_input_when_run_main_then_succeeds(self):
```


### What to test?

For each class __at least__ validate:
- Its initialization.
- Its expected failure when a `raise` statement is present.
- Its expected field values for a default initialization.

For each "public" method __at_least__ validate:
- Its expected result when all valid arguments are given.
- Its expected failure when a `raise` statement is present.

In addition, integration and system tests are highly encouraged.

### How to write a test?

To create a test, follow the [vrtool code standards](#code-standards) and divide your test following the principles of _"Given an initial situation, When something happens, Then expectation is met".

This can be represented in the content and/or the name, but its entirely up to the contributor. Please keep in mind that the test name should remain short yet comprehensive. An example could be:
```python
def given_valid_input_when_run_assessment_then_succeeds(self):
    # 1. Given / Define initial expectations.
    ...
    
    # 2. When / Run test.
    ...
    
    # 3. Then / Verify final expectations.
    ...
```


## Adding documentation

Besides the in-code documentation via docstrings, adding new features or modifying some pieces of code may require the addition of more documentation. Possible options are:

1. Create a `README.md` at the module's level. For instance in `vrtool.defaults` you will find one specific file describing the purpose of said module.
2. Create an extensive markdown file in the [docs](.) directory, like this `CONTRIBUTING.md` file.
3. Create / update the available tutorials. Some code modifications might have an impact on the existing documentation. Consider either updating, extending or creating new one to ensure "new" contributors can adapt easily enough.

We encourage as well contributors to add and extend our [VRTOOL wiki pages](https://github.com/Deltares/Veiligheidsrendement/wiki).