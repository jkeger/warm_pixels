import os
from abc import ABC, abstractmethod

import numpy as np

import warm_pixels.hst_functions.plot
from warm_pixels import hst_functions as fu
from warm_pixels import hst_utilities as ut
from warm_pixels import hst_utilities as ut
from warm_pixels.hst_data import Dataset
from warm_pixels.hst_functions import plot
from warm_pixels.hst_utilities import output_path
from warm_pixels.pixel_lines import PixelLine, PixelLineCollection
from warm_pixels.warm_pixels import find_dataset_warm_pixels


class AbstractProcess(ABC):
    def __init__(
            self,
            dataset,
            warm_pixels,
    ):
        self.dataset = dataset
        self.warm_pixels = warm_pixels

    def run(self):
        for quadrant in self.warm_pixels.all_quadrants:
            self.process_quadrant(quadrant)

        self.plot_distributions()

        # Stack warm pixels in each image quadrant or combined quadrants
        for quadrants in self.warm_pixels.quadrant_sets:
            self.stack_warm_pixels(quadrants)

    def plot_distributions(self):
        # Plot distributions of warm pixels in the set
        if self.warm_pixels.need_to_make_file(
                self.dataset.plotted_distributions(self.warm_pixels.all_quadrants),
        ):
            print("  Distributions of warm pixels...", end=" ", flush=True)
            fu.plot_warm_pixel_distributions(
                self.dataset,
                self.warm_pixels.all_quadrants,
                save_path=self.dataset.plotted_distributions(self.warm_pixels.all_quadrants),
            )

    def stack_warm_pixels(self, quadrants):
        # Stack in bins
        if self.warm_pixels.need_to_make_file(
                self.dataset.saved_stacked_lines(quadrants, self.warm_pixels.use_corrected),
        ):
            print(
                f"  Stack warm pixel trails ({''.join(quadrants)})...",
                end=" ",
                flush=True,
            )
            fu.stack_dataset_warm_pixels(self.dataset, quadrants, self.warm_pixels.use_corrected)

        # Plot stacked lines
        if not self.warm_pixels.use_corrected and self.warm_pixels.need_to_make_file(
                self.dataset.plotted_stacked_trails(
                    quadrants,
                    self.warm_pixels.use_corrected
                ),
        ):
            print(
                f"  Plot stacked trails ({''.join(quadrants)})...",
                end=" ",
                flush=True,
            )
            fu.plot_stacked_trails(
                self.dataset,
                quadrants,
                use_corrected=self.warm_pixels.use_corrected,
                save_path=self.dataset.plotted_stacked_trails(
                    quadrants, self.warm_pixels.use_corrected
                ),
            )

    @abstractmethod
    def process_quadrant(self, quadrant):
        pass


class RawProcess(AbstractProcess):
    def process_quadrant(self, quadrant):
        # Find possible warm pixels in each image
        if self.warm_pixels.need_to_make_file(
                self.dataset.saved_lines(quadrant),
        ):
            print(
                f"  Find possible warm pixels ({quadrant})...",
                end=" ",
                flush=True,
            )
            find_dataset_warm_pixels(self.dataset, quadrant)

        # Find consistent warm pixels in the set
        if self.warm_pixels.need_to_make_file(
                self.dataset.saved_consistent_lines(quadrant),
        ):
            print(
                f"  Consistent warm pixels ({quadrant})...",
                end=" ",
                flush=True,
            )
            fu.find_consistent_warm_pixels(
                self.dataset,
                quadrant,
                flux_min=ut.flux_bins[0],
                flux_max=ut.flux_bins[-1],
            )


class CorrectedProcess(AbstractProcess):
    def __init__(
            self,
            raw_process
    ):
        super().__init__(
            dataset=raw_process.dataset.corrected(),
            warm_pixels=raw_process.warm_pixels,
        )
        self.raw_process = raw_process

    def process_quadrant(self, quadrant):
        self.raw_process.process_quadrant(quadrant)

        # Extract from corrected images with CTI removed
        if self.warm_pixels.need_to_make_file(
                self.dataset.saved_consistent_lines(quadrant, use_corrected=True),
        ):
            print(f"  Extract CTI-removed warm pixels ({quadrant})...")
            self.extract_consistent_warm_pixels_corrected(quadrant)

    def extract_consistent_warm_pixels_corrected(self, quadrant):
        """Extract the corresponding warm pixels from the corrected images with CTI
        removed, in the same locations as the orignal consistent warm pixels.

        Parameters
        ----------
        dataset : Dataset
            The dataset object with a list of image file paths and metadata.

        quadrant : str (opt.)
            The quadrant (A, B, C, D) of the image to load.

        Saves
        -----
        warm_pixels_cor : PixelLineCollection
            The set of consistent warm pixel trails, saved to
            dataset.saved_consistent_lines(use_corrected=True).
        """
        # Load original warm pixels for the whole dataset
        warm_pixels = PixelLineCollection()
        warm_pixels.load(self.raw_process.dataset.saved_consistent_lines(quadrant))

        # Corrected images
        warm_pixels_cor = PixelLineCollection()
        for i, image in enumerate(self.dataset):
            image_name = image.name
            print(
                f"\r    {image_name}_cor_{quadrant} ({i + 1} of {len(self.dataset)}) ",
                end="",
                flush=True,
            )

            # Load the image
            array = image.load_quadrant(quadrant)

            # Select consistent warm pixels found from this image
            image_name_q = image_name + "_%s" % quadrant
            sel = np.where(warm_pixels.origins == image_name_q)[0]
            for i in sel:
                line = warm_pixels.lines[i]
                row, column = line.location

                # Copy the original metadata but take the data from the corrected image
                warm_pixels_cor.append(
                    PixelLine(
                        data=array[
                             row - ut.trail_length: row + ut.trail_length + 1, column
                             ],
                        origin=line.origin,
                        location=line.location,
                        date=line.date,
                        background=line.background,
                    )
                )

        print("Extracted %d lines" % warm_pixels_cor.n_lines)

        # Save
        warm_pixels_cor.save(self.dataset.saved_consistent_lines(quadrant, use_corrected=True))


class WarmPixels:
    def __init__(
            self,
            datasets,
            quadrants,
            overwrite=False,
            prep_density=False,
            use_corrected=False,
            plot_density=False,
    ):
        self.datasets = datasets
        self.quadrants = quadrants
        self.overwrite = overwrite
        self.prep_density = prep_density
        self.use_corrected = use_corrected
        self.plot_density = plot_density

        # TODO: list name was originally the input...
        self.list_name = "test"

    @property
    def quadrant_sets(self):
        return [[q for q in qs] for qs in self.quadrants.split("_")]

    def need_to_make_file(self, filename):
        if self.overwrite:
            return True
        return not os.path.exists(filename)

    @property
    def all_quadrants(self):
        # All quadrants, ignoring subsets
        return [q for qs in self.quadrant_sets for q in qs]

    def main(self):
        # ========
        # Create directories to contain output plots
        # ========
        os.makedirs(ut.output_path / "stacked_trail_plots", exist_ok=True)
        os.makedirs(ut.output_path / "plotted_distributions", exist_ok=True)

        # ========
        # Find and stack warm pixels in each dataset
        # ========
        for i_dataset, dataset in enumerate(self.datasets):
            print(
                f'Dataset "{dataset.name}" '
                f'({i_dataset + 1} of {len(self.datasets)}, '
                f'{len(dataset)} images, "{self.quadrants}")'
            )
            process = RawProcess(
                dataset,
                self
            )
            if self.use_corrected:
                process = CorrectedProcess(process)
            process.run()

        # ========
        # Compiled results from all datasets
        # ========
        # Fit and save the total trap densities
        if self.prep_density:
            # In each image quadrant or combined quadrants
            for quadrants in self.quadrant_sets:
                print(
                    f"Fit total trap densities ({''.join(quadrants)})...",
                    end=" ",
                    flush=True,
                )
                fu.fit_total_trap_densities(
                    self.datasets, self.list_name, quadrants, self.use_corrected
                )

        # Plot the trap density evolution
        if self.plot_density:
            print("Plot trap density evolution...", end=" ", flush=True)
            plot.plot_trap_density_evol(
                self.list_name, self.quadrant_sets, do_sunspots=True, use_corrected=self.use_corrected
            )
