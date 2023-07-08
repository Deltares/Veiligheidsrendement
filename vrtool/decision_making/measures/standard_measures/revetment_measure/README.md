# Revetment Measure.

A revetment measure computes the `Beta` and `Probability of Failure` based on a `Beta Target` (vector) and `Transition Level` (vector) for a given `Dike Section`. 

This module contains different classes representing differnt aspects of the aforementioned computation:

- `RevetmentMeasureData`: Used to represent the new revetment properties of a slope from the `revetment mechanism`.
- `RevetmentMeasureResult`: Data structure used to represent the measure's year (`measure_year`), Beta target (`beta_target`), Beta combined (`beta_combined`), transition level (`transition_level`), measure cost (`cost`) and the related `revetment_measures` (list of `RevetmentMeasureData`).
- `RevetmentMeasureResultBuilder`: Generates a `RevetmentMeasureResult` (or a list of `RevetmentMeasureData`).
- `RevetmentMeasureSectionReliability`: Data object responsible to wrap the `SectionReliability` for a given `beta_target` and `transition_level` with its related `cost`. It also implements the `MeasureResultCollectionProtocol` so it defines how to represent itself in the `solutions.solutions_to_dataframe` method.
- `RevetmentMeasureResultCollection`: A specialization of the `MeasureResultCollectionProtocol` to wrap all the `RevetmentMeasureSectionReliability` and define how to represent their output during the `solutions.solutions_to_dataframe` method.
- `RevetmentMeasure`: Controller implementing the `MeasureProtocol` which contains all the imported parameters from the database. It is responsible to map for each combination of `beta_target` and `transition_level` the matching `SectionReliability`.