# Contribution guidelines

At the moment access to this repository is restricted. To be able to contribute please contact the administrator(s) of this project. In case you already have access, please adhere to the code of conduct consisting of the following:

* [Use of JIRA board for issue traceability](#jira-board).
* [Create branches for any contribution](#creating-new-branches).
* Usage of code formatters and standards.
* Writing tests
* Writing documentation.
* Code review
* Merging through pull-requests.

## JIRA Board.

This repository is developed along the usage of a [Deltares JIRA project](https://issuetracker.deltares.nl/secure/RapidBoard.jspa?rapidView=810&projectKey=VRTOOL&view=planning&issueLimit=100) to track the requests for new developments (features, documentation, bug fixes, ...) or related tasks (Continuous Integration, design choices and so on). If you lack access to said board, please contact the project administrator(s).

### How to use the board? 
We usually work in sprints. Issues are prioritized accordingly and therefore only those within a sprint should be picked. A contributor is only expected to be actively working on one issue at the time, although they can also participate by doing reviews or testing the work done by others.

### When should issues be created?
An issue can be created for multiple reasons.

* Bug: an error or malfunctioning in the code is found and needs a fix.
* New feature: a new development is needed to fulfill a (non) functional requirement. This usually consists of writing entirely new code.
* Improvement: same as new feature although in this case consists of partially re-writing the existing code to enhance its performance and functionalities.
* Task: usually related to non-development related actions, such as setting up Continuous Integration platforms or consulting over the project.
* Documentation: anything related to writing or updating the project's documentation.

When creating an issue please try to describe it as best as possible including:
- Which version of the  `vrtool` is affected by it.
- Which version of the `vrtool`is expected to fulfill it.
- What is the current situation. What is happening?
- What is the desired situation. What needs to be done?
- When is the issue considered as completed (completion criteria).

In addition to the above points:
- For code developments (bug / new / improvement) please also add test data whenever possible.
- For bugs or improvements, describe the exact steps (and system used) needed to reproduce the current situation.

## Creating new branches

New branches are encouraged based on the _dare to share_  principle, this allows the rest of contributors to be aware of your changes as soon as possible. However, please do not forget to create a related issue as described in the [above section](#when-should-issues-be-created).

All created branches should adhere to the pattern `prefix\VRTOOL-##_title_in_snake_case`. Where:

* `prefix`, is a descriptive string representing the type of issue. The current accepted prefixes are the following:
    - __main__: This prefix is not to be used as containts the latest (greatest) changes. To contribute into this branch you will have to do a pull-request from your own branch.
    - __feature/*__: for issues tagged as "new feature" or "improvement".
    - __bug/*__:  for issues tagged as "bug".
    - __poc/*__: for research and development issues outside an official sprint.
    - __chore/*__: for issues tagged as "task" or "improvement" that are trivial such as cleaning up code or modifying a few tests. 
    - __docs/*__: for issues tagged as "documentation".
* `VRTOOL-##`, is the issue number that can be found in its own page (for instance https://issuetracker.deltares.nl/browse/VRTOOL-71) would be `VRTOOL-71`.
* and `title_in_snake_case` is a short abstract of the title (they can be often too long) in the snake case format, issue `VRTOOL-71 Create collaboration How to.` should be `_create_collaboration_how_to`.

So, as an example the documentation issue `VRTOOL-71 Create collaboration How to` would be carried out in a branch named as `docs/VRTOOL-71_create_collaboration_how_to`.

## Code formatters and styling.

To ensure the correct normalization of the code in the `main` branch, we actively run a [Github workflow](../.github/workflows/normalize_code.yml) that formats the new additions with `black` and `isort`.

You can locally run these formatters with `poetry run isort . && poetry run black`.