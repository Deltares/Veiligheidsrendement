# Run workflows module

This module intends to contain all supported workflows that can be either run through CLI commands or through basic sandbox usage.

New workflows should adhere to the `VrToolRunProtocol`. At the same time, a concrete `VrToolRunResultProtocol` definition for said new workflow should be created to represent its generated run data (could be None). It is recommended to name them in a similar way to the rest, following the format "run__name_of_the_workflow__.py"