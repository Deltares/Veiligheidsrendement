# Measures

This module contains all the available measures to evaluate the effect of a failure mechanism in a given `DikeSection`.

Any new measure should implement the `MeasureProtocol` located in `measure_protocol.py`. The evaluation of a measure will be done within the protocol's method `evaluate_measure`. A new measure's result should be written to a specialization of the `MeasureResultCollectionProtocol` to delegate there, and not elsewhere, its reliability output representation.

We differenciate between two types of measures:

- Standard measures, predefined set of measures which we support and have explicit logic for. They are contained within the submodule `standard_measures` and consist of:
    - Diaphragm wall (`DiaphragmWallMeasure`),
    - Anchored sheetpile (`AnchoredSheetpileMeasure`), a subtype of `DiaphragmWallMeasure`,
    - Soil reinforcement (`SoilReinforcementMeasure`),
    - Stability screen (`StabilityScreenMeasure`),
    - Vertical piping solution measure (`VerticalPipingSolutionMeasure`), this measure is internally (not exposed via the `MeasureTypeEnum`) divided depending on the `cover_layer_thickness` value into:
        - __Coarse Sand Barrier__ (Dutch abbreviation: __GZB__) - `CoarseSandBarrierMeasureCalculator`,
        - __Vertical Geotextile__ (Dutch abbreviation: __VZG__) - `VerticalGeotextileMeasureCalculator`,
        - __Heavescreen__ - `HeavescreenMeasureCalculator`,
    - Revetment measure (`RevetmentMeasure`)

- Custom measure, any measure not explicitely defined by us which can be addded to an assessment. The evaluation of these measures is implicitly done during the import of these measures, so the `evaluate_measure` will not do anything.