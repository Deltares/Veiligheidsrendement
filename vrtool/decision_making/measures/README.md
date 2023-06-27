# Measures

This module contains all the available measures to evaluate the effect of a failure mechanism in a given `DikeSection`.

Any new measure should implement the `MeasureProtocol` located in `measure_protocol.py`. The evaluation of a measure will be done within the protocol's method `evaluate_measure`.

We differenciate between two types of measures:

- Standard measures, predefined set of measures which we support and have explicit logic for. They consist of:
    - Diaphragm wall (`DiaphragmWallMeasure`),
    - Soil reinforcement (`SoilReinforcementMeasure`),
    - Stability screen (`StabilityScreenMeasure`),
    - Vertical geotextile measure (`VerticalGeotextileMeasure`),
    - Revetment measure (`RevetmentMeasure`)

- Custom measure, any measure not explicitely defined by us which can be addded to an assessment.