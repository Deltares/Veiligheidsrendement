# Combined measures

This subproject contains the different classes that represent all ways of combining measures (`MeasureAsInputProtocol`).

A `CombinedMeasure` is actually represented by the generalization`CombinedMeasureBase` and it 
has the following specializations:

- `ShCombinedMeasure`, when the `primary` measure to combine is of type `ShMeasure`. It also has an optional `secondary` measure that can be of any type.
- `SgCombinedMeasure`, when the `primary` measure to combine is of type `SgMeasure`, its `lcc` is always based on a `base_cost` equal to `0`. It also has an optional `secondary` measure that can be of any type.
- `ShSgCombinedMeasure`, when the `primary` measure to combine is of type `ShSgMeasure`, its `lcc`is always based on a `base_cost` equal to `0`. It also has two optional secondary measures: `sh_secondary` with a `ShMeasure` and `sg_secondary` with a `SgMeasure`.

## Creation of combined measures

All types of `CombinedMeasureBase` can be created as any other `dataclass`. However, there is a factory (`CombinedMeasureFactory`) available to speed up the creation process.

Both `ShCombinedMeasure` and `SgCombinedMeasure`  are created from the `CombineMeasuresController`, however the `ShSgCombinedMeasure` is only created at the `AggregateCombinationsController` as it requires an already existing collection of `CombinedMeasureBase` instances (usually of both `ShCombinedMeasure` and `SgCombinedMeasure`). This is the reason why the `CombinedMeasureFactory.from_input` does not support the creation of `ShSgCombinedMeasure`.