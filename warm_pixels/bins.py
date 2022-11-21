from typing import Optional, Iterable

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
            values: Iterable[float],
            n_bins: int,
            min_value: Optional[float] = None,
            max_value: Optional[float] = None,
            scale: str = "linear"
    ) -> "Bins":
        """
        Constructs a collection of bins either between the minimum and maximum
        values from some collection or between explicitly set minimum and maximum
        values.

        Parameters
        ----------
        values
            A collection of values
        n_bins
            The number of bins
        min_value
            The minimum bound of the lowest bin
        max_value
            The maximum bound of the highest bin
        scale
            The scale (linear or log)

        Returns
        -------
        An object representing value bins
        """
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

    def index(self, value):
        if value > self.max:
            raise IndexError(
                f"Value {value} is greater than upper bound of max bin {self.max}"
            )
        return np.digitize(value, self[:-1]) - 1
