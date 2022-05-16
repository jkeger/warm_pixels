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
        return self.row_bins.number

    @property
    def n_flux_bins(self):
        return self.flux_bins.number

    @property
    def n_background_bins(self):
        return self.background_bins.number
