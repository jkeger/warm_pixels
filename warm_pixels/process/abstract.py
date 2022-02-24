import os
from abc import ABC, abstractmethod

from warm_pixels import hst_functions as fu, PixelLineCollection


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
            filename = self.dataset.saved_consistent_lines(quadrant)
            if self.need_to_make_file(filename):
                consistent_lines = self.consistent_lines_for(quadrant)
                consistent_lines.save(filename)
            else:
                consistent_lines = PixelLineCollection()
                consistent_lines.load(filename)

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
    def consistent_lines_for(self, quadrant):
        pass
