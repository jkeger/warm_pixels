import numpy as np


class Bins:
    def __init__(self, bins):
        self.bins = bins

    @property
    def number(self):
        return len(self.bins) - 1

    @property
    def min(self):
        return self.bins[0]

    @property
    def max(self):
        return self.bins[-1]

    def __getitem__(self, item):
        return self.bins[item]

    @classmethod
    def from_values(
            cls,
            values,
            n_bins,
            min_value=None,
            max_value=None,
            scale="linear"
    ):
        min_value = min_value or min(values)
        max_value = max_value or max(values)

        # Bin lower edge values
        if scale == "linear":
            return Bins(
                np.linspace(
                    min_value,
                    max_value,
                    n_bins + 1
                )
            )
        elif scale == "log":
            return Bins(
                np.logspace(
                    np.log10(min_value),
                    np.log10(max_value),
                    n_bins
                )
            )
        else:
            raise ValueError(
                "Scale must be linear or log"
            )

    @property
    def is_single(self):
        return self.max == self.min or self.number == 1

    def indices(self, values):
        indices = np.digitize(values, self[:-1]) - 1
        # Flag if above the max
        indices[values > self.max] = -1
        return indices
