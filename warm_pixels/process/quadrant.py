import numpy as np

from warm_pixels import hst_utilities as ut
from warm_pixels.pixel_lines import PixelLine, PixelLineCollection
from warm_pixels.warm_pixels import find_dataset_warm_pixels


class Quadrant:
    def __init__(self, quadrant, dataset):
        self.quadrant = quadrant
        self.dataset = dataset

    def __str__(self):
        return self.quadrant

    def consistent_lines(self):
        warm_pixels = find_dataset_warm_pixels(
            self.dataset,
            self.quadrant
        )

        return warm_pixels.consistent(
            flux_min=ut.flux_bins[0],
            flux_max=ut.flux_bins[-1],
        )


class CorrectedQuadrant(Quadrant):
    def consistent_lines(self):
        """Extract the corresponding warm pixels from the corrected images with CTI
        removed, in the same locations as the orignal consistent warm pixels.
        """
        warm_pixels = super().consistent_lines()

        # Corrected images
        warm_pixels_cor = PixelLineCollection()
        for i, image in enumerate(self.dataset):
            image_name = image.name
            print(
                f"\r    {image_name}_cor_{self.quadrant} ({i + 1} of {len(self.dataset)}) ",
                end="",
                flush=True,
            )

            # Load the image
            array = image.load_quadrant(self.quadrant)

            # Select consistent warm pixels found from this image
            image_name_q = f"{image_name}_{self.quadrant}"
            sel = np.where(warm_pixels.origins == image_name_q)[0]
            for j in sel:
                line = warm_pixels.lines[j]
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

        print(f"Extracted {warm_pixels_cor.n_lines} lines")

        return warm_pixels_cor
