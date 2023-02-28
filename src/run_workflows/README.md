# Run workflows module

This module intends to contain all supported workflows that can be either run through CLI commands or through basic sandbox usage.

New workflows should adhere to the `VrToolRunProtocol`. At the same time, a concrete `VrToolRunResultProtocol` definition for said new workflow should be created to represent its generated run data (could be None). Consider creating each workflow and its realted result class in their own submodule.