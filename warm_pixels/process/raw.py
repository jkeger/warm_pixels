from warm_pixels import hst_functions as fu
from .abstract import AbstractProcess
from .quadrant import Quadrant


class RawProcess(AbstractProcess):
    def consistent_lines_for(self, quadrant):
        return Quadrant(
            quadrant=quadrant,
            dataset=self.dataset
        ).consistent_lines()

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
                    process=self,
                    use_corrected=False,
                    save_path=filename,
                    quadrants=group,
                )
