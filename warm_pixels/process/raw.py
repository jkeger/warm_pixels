from warm_pixels import hst_functions as fu, PixelLineCollection
from warm_pixels import hst_utilities as ut
from warm_pixels.warm_pixels import find_dataset_warm_pixels
from .abstract import AbstractProcess


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
            warm_pixels = find_dataset_warm_pixels(self.dataset, quadrant)
        else:
            warm_pixels = PixelLineCollection()
            warm_pixels.load(self.dataset.saved_lines(quadrant))

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
                warm_pixels=warm_pixels,
                flux_min=ut.flux_bins[0],
                flux_max=ut.flux_bins[-1],
            )

    def stack_warm_pixels(self, quadrants):
        super().stack_warm_pixels(quadrants)
        # Plot stacked lines
        # TODO: this runs for corrected
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
