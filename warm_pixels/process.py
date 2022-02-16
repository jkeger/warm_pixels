import os
from abc import ABC, abstractmethod

import numpy as np

from warm_pixels import hst_functions as fu
from warm_pixels import hst_utilities as ut
from warm_pixels.pixel_lines import PixelLine, PixelLineCollection
from warm_pixels.warm_pixels import find_dataset_warm_pixels


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

    def run(self):
        for quadrant in self.quadrants:
            self.process_quadrant(quadrant)

        self.plot_distributions()

        # Stack warm pixels in each image quadrant or combined quadrants
        for quadrants in self.quadrants.groups:
            self.stack_warm_pixels(quadrants)

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
            fu.stack_dataset_warm_pixels(self.dataset, quadrants)

    @abstractmethod
    def process_quadrant(self, quadrant):
        pass


class RawProcess(AbstractProcess):
    def process_quadrant(self, quadrant):
        # Find possible warm pixels in each image
        if self.need_to_make_file(
                self.dataset.saved_lines(quadrant),
        ):
            print(
                f"  Find possible warm pixels ({quadrant})...",
                end=" ",
                flush=True,
            )
            find_dataset_warm_pixels(self.dataset, quadrant)

        # Find consistent warm pixels in the set
        if self.need_to_make_file(
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

    def stack_warm_pixels(self, quadrants):
        super().stack_warm_pixels(quadrants)
        # Plot stacked lines
        if self.need_to_make_file(
                self.dataset.plotted_stacked_trails(
                    quadrants,
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
                use_corrected=False,
                save_path=self.dataset.plotted_stacked_trails(
                    quadrants
                ),
            )


class CorrectedProcess(AbstractProcess):
    def __init__(
            self,
            raw_process
    ):
        super().__init__(
            dataset=raw_process.dataset.corrected(),
            overwrite=raw_process.overwrite,
            quadrants=raw_process.quadrants,
        )
        self.raw_process = raw_process

    def process_quadrant(self, quadrant):
        self.raw_process.process_quadrant(quadrant)

        # Extract from corrected images with CTI removed
        if self.need_to_make_file(
                self.dataset.saved_consistent_lines(quadrant),
        ):
            print(f"  Extract CTI-removed warm pixels ({quadrant})...")
            self.extract_consistent_warm_pixels_corrected(quadrant)

    def extract_consistent_warm_pixels_corrected(self, quadrant):
        """Extract the corresponding warm pixels from the corrected images with CTI
        removed, in the same locations as the orignal consistent warm pixels.

        Parameters
        ----------
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
            image_name_q = f"{image_name}_{quadrant}"
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
        warm_pixels_cor.save(self.dataset.saved_consistent_lines(quadrant))
