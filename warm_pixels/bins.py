import numpy as np


class Bins:
    def __init__(self, bins):
        self.bins = bins

    def number(self):
        return len(self.bins) - 1

    @property
    def min(self):
        return self.bins[0]

    @property
    def max(self):
        return self.bins[-1]

    @classmethod
    def from_values(
            cls,
            values,
            n_row_bins,
            row_min=None,
            row_max=None,
            row_scale="linear"
    ):
        row_min = row_min or min(values)
        row_max = row_max or max(values)

        # Bin lower edge values
        if row_scale == "linear":
            return Bins(
                np.linspace(row_min, row_max, n_row_bins + 1)
            )
        elif row_scale == "log":
            return Bins(
                np.logspace(
                    np.log10(row_min),
                    np.log10(row_max),
                    n_row_bins + 1
                )
            )
        else:
            raise ValueError(
                "Row scale must be linear or log"
            )
