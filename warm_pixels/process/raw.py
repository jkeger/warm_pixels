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

    def plot(self):
        super().plot()
        for group in self.quadrants.groups:
            filename = self.dataset.plotted_stacked_trails(
                group,
            )
            if self.need_to_make_file(
                    filename
            ):
                fu.plot_stacked_trails(
                    dataset=self.dataset,
                    use_corrected=False,
                    save_path=filename,
                    quadrants=group,
                    stacked_lines=self.stacked_lines_for_group(group)
                )
