# Vertical piping solutions

The vertical piping solution is dependent on the thickness of the cover layer, d_cover.

Depending on the depth of the cover layer a variation three different vertical piping solutions are possible, each solution has the same effect on piping, but each has a different cost:

- The __Course Sand Barrier__ (Dutch abbreviation: __GZB__) is applied when `d_cover <2m`. It reduces the `pf_piping` with a factor `1000` and has a price of `1700€/m`.
- The __Vertical Geotextile__ (Dutch abbreviation: __VZG__) is applied when `2m <= d_cover < 4m`. It reduces the `pf_piping` with a factor `1000` and has a price of `1700€/m`.
- The __heavescreen__: applied when `4m <= d_cover < 6m`. It reduces the `pf_piping` with a factor `1000`. The price is expressed per m2, so we need to calculate the vertical length of the screen. The assumption is that it should go 6m below the cover_layer, so `l_screen = d_cover + 6m`. The unit cost is assumed to be lower than that from the unanchored sheetpile: `400€/m2`.

If `d_cover > 6m`, the probability of piping should be assumed minimal.