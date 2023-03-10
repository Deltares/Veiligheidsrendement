# Failure mechanisms module

This module is intended to contain all failure mechanisms that can be used throughout the module. 

There is a generic input that can be used to host information for every failure mechanism. Implementations for specific failure mechanisms are implemented within the individual submodules and can contain (but is not limited to):

* Definitions for failure mechanism specific input.
* A general function library of a failure mechanism which is used througout the application and is not limited to calculating the reliability and the safety factor.
* An implementation to calculate the reliability and the safety factor.