import numpy as np


class combinFunctions:
    def combine_probabilities(
        self, prob_of_failure: dict[str, np.array], selection
    ) -> np.array:

        cnt = 0
        for mechanism in selection:
            if mechanism in prob_of_failure:
                cnt += 1
                p = prob_of_failure.get(mechanism, 0)
                if cnt == 1:
                    product = 1 - p
                else:
                    product = np.multiply(product, 1 - p)

        if cnt == 1:
            # p is in this case almost equal to 1 - product, but p is more accurate
            return p
        else:
            return 1 - product
