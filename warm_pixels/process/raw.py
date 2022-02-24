from warm_pixels import hst_functions as fu
from warm_pixels import hst_utilities as ut
from warm_pixels.warm_pixels import find_dataset_warm_pixels
from .abstract import AbstractProcess


class RawProcess(AbstractProcess):
    def consistent_lines_for(self, quadrant):
        warm_pixels = find_dataset_warm_pixels(
            self.dataset,
            quadrant
        )

        return warm_pixels.consistent(
            flux_min=ut.flux_bins[0],
            flux_max=ut.flux_bins[-1],
        )

    def run(self):
        super().run()
        for quadrants in self.quadrants.groups:
            self.plot_stacked_trails(quadrants)

    def plot_stacked_trails(self, quadrants):
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
