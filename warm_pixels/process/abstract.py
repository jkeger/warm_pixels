import os
from abc import ABC, abstractmethod

from warm_pixels import hst_functions as fu, PixelLineCollection
from warm_pixels import hst_utilities as ut


class AbstractProcess(ABC):
    def __init__(
            self,
            dataset,
            quadrants,
            overwrite,
    ):
        self.dataset = dataset
        self.overwrite = overwrite
        self.quadrants = quadrants

    def need_to_make_file(self, filename):
        if self.overwrite:
            return True
        return not os.path.exists(filename)

    def consistent_lines_for_quadrant(
            self,
            quadrant
    ):
        filename = self.dataset.saved_consistent_lines(quadrant)
        if self.need_to_make_file(filename):
            consistent_lines = self.consistent_lines_for(quadrant)
            consistent_lines.save(filename)
            return consistent_lines

        return PixelLineCollection.load(filename)

    def run(self):
        # Stack warm pixels in each image quadrant or combined quadrants
        for quadrants in self.quadrants.groups:
            self.stack_warm_pixels(quadrants)

        self.plot_distributions()

    def stack_warm_pixels(self, quadrants):
        # Stack in bins
        if self.need_to_make_file(
                self.dataset.saved_stacked_lines(quadrants),
        ):
            print(
                f"  Stack warm pixel trails ({''.join(quadrants)})...",
                end=" ",
                flush=True,
            )
            self.stack_dataset_warm_pixels(quadrants)

    def stack_dataset_warm_pixels(self, quadrants):
        """Stack a set of premade warm pixel trails into bins.

        find_dataset_warm_pixels() and find_consistent_warm_pixels() must first be
        run for the dataset.

        Parameters
        ----------
        quadrants : [str]
            The list of quadrants (A, B, C, D) of the images to load, combined
            together if more than one provided.
        """
        warm_pixels = sum(
            self.consistent_lines_for_quadrant(
                quadrant
            )
            for quadrant in quadrants
        )

        # Stack the lines in bins by distance from readout and total flux
        stacked_lines = warm_pixels.generate_stacked_lines_from_bins(
            n_row_bins=ut.n_row_bins,
            flux_bins=ut.flux_bins,
            n_background_bins=ut.n_background_bins,
        )
        print(
            "Stacked lines in %d bins"
            % (ut.n_row_bins * ut.n_flux_bins * ut.n_background_bins)
        )

        # Save
        stacked_lines.save(self.dataset.saved_stacked_lines(quadrants))

    def plot_distributions(self):
        # Plot distributions of warm pixels in the set
        if self.need_to_make_file(
                self.dataset.plotted_distributions(self.quadrants),
        ):
            print("  Distributions of warm pixels...", end=" ", flush=True)
            fu.plot_warm_pixel_distributions(
                self.dataset,
                self.quadrants,
                save_path=self.dataset.plotted_distributions(self.quadrants),
            )

    @abstractmethod
    def consistent_lines_for(self, quadrant):
        pass
