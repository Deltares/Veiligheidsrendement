# Failure mechanisms module

This module is intended to contain all failure mechanisms that can be used throughout the module. 

There is a generic input that can be used to host information for every failure mechanism. Implementations for specific failure mechanisms are implemented within the individual submodules and can contain (but is not limited to):

* Definitions for failure mechanism specific input.
* A general function library of a failure mechanism which is used througout the application and is not limited to calculating the reliability and the safety factor.
* An implementation of the `FailureMechanismCalculatorProtocol` to calculate the reliability and the safety factor.

The idea for each failure mechanism module is as follows:

* A module has a specific input which unpacks the `MechanismInput` into the elements and properties that are required for the failure mechanism implementation to calculate the reliability and the safety factor.
* The specific input of the previous point is subsequently used by the implementation of the `FailureMechanismCalculatorProtocol` to calculate the reliability and the safety factor. Note that the calculator can be dependent on other parameters as well.
* In case certain functionality is not only used in the failure mechanism implementation, but also at other parts of the application, a failure mechanism specific function library can be introduced.