# Vertical piping measures

The vertical piping (measure) solution is dependent on the thickness of the cover layer, `cover_layer_thickness`.

Depending on the depth of the cover layer a variation three different vertical piping solutions are possible, each solution has the same effect on piping, but each has a different cost:

- The __Course Sand Barrier__ (Dutch abbreviation: __GZB__) - `CourseSandBarrierMeasureCalculator`: applied when `cover_layer_thickness <2m`. It reduces the `pf_piping` with a factor `1000` and has a price of `1700€/m`.
- The __Vertical Geotextile__ (Dutch abbreviation: __VZG__) - `VerticalGeotextileMeasureCalculator`: applied when `2m <= cover_layer_thickness < 4m`. It reduces the `pf_piping` with a factor `1000` and has a price of `1700€/m`.
- The __Heavescreen__ - `HeavescreenMeasureCalculator`: applied when `4m <= cover_layer_thickness < 6m`. It reduces the `pf_piping` with a factor `1000`. The price is expressed per m2, so we need to calculate the vertical length of the screen. The assumption is that it should go 6m below the cover_layer, so `l_screen = cover_layer_thickness + 6m`. The unit cost is assumed to be lower than that from the unanchored sheetpile: `400€/m2`.

If `cover_layer_thickness > 6m`, the probability of piping should be assumed minimal.