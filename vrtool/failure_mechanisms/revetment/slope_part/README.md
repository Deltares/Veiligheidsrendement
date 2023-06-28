# Slope Parts module

This module consists of all possible definitions of a slope part. A slope part is represented by the `SlopePartProtocol`, and currently we support the following slope part types:

- Asphalt (`AsphaltSlopePart`),
- Grass (`GrassSlopePart`),
- Stone (`StoneSlopePart`)

In addition, when a slope part is modified we can represent the modified slope part with the `ModifiedSlopePart` class, which contains two properties, one for the original slope part (`previous_slope_part`) and other for the new calculated slope (`modified_slope_part`).