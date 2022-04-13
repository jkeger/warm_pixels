import os
from pathlib import Path

import numpy as np

from .collection import PixelLineCollection


class StackedPixelLineCollection(PixelLineCollection):
    def __init__(
            self,
            lines,
            row_bins,
            flux_bins,
            date_bins,
            background_bins,
    ):
        super().__init__(lines)
        self.row_bins = row_bins
        self.flux_bins = flux_bins
        self.date_bins = date_bins
        self.background_bins = background_bins

    @property
    def n_row_bins(self):
        return len(self.row_bins) - 1

    @property
    def n_flux_bins(self):
        return len(self.flux_bins) - 1

    @property
    def n_background_bins(self):
        return len(self.background_bins) - 1

    def save(self, filename):
        path = Path(filename)
        os.makedirs(
            path,
            exist_ok=True
        )
        super().save(
            path / "lines.pickle"
        )
        np.savez(
            str(path / "info.npz"),
            self.row_bins,
            self.flux_bins,
            self.date_bins,
            self.background_bins,
        )

    @classmethod
    def load(cls, filename):
        path = Path(filename)
        stacked_lines = PixelLineCollection.load(
            path / "lines.pickle"
        )
        npzfile = np.load(str(path / "info.npz"))
        row_bins, flux_bins, date_bins, background_bins = [
            npzfile[var] for var in npzfile.files
        ]
        return StackedPixelLineCollection(
            stacked_lines.lines,
            row_bins,
            flux_bins,
            date_bins,
            background_bins,
        )
