import logging

from warm_pixels import hst_utilities as ut
from .stacked_trails import plot_stacked_trails as stacked_trails
from .trap_density import plot_trap_density_evol as trap_density_evol
from .warm_pixels import plot_warm_pixel_distributions as warm_pixel_distributions
from .warm_pixels import plot_warm_pixels as warm_pixels
from ..output import AbstractOutput, _check_path

logger = logging.getLogger(__name__)


class Plot(AbstractOutput):
    def warm_pixels(self):
        """
        Plot warm pixels for each quadrant of each dataset
        """
        for dataset in self._warm_pixels.datasets:
            self.plot_all_warm_pixels(dataset)

    def plot_all_warm_pixels(self, dataset):
        for quadrant in self._warm_pixels.all_quadrants_string:
            quadrant = dataset.quadrant(quadrant)
            for image_quadrant in quadrant.image_quadrants:
                filename = ut.output_path / dataset.name / f"{image_quadrant.name}.png"
                if _check_path(filename):
                    continue

                print(f"Plotting warm pixels for {dataset}/{image_quadrant}")

                # Plot
                warm_pixels(
                    image_quadrant,
                    save_path=filename,
                )

    def warm_pixel_distributions(self):
        """
        Plot a histogram of the distribution of warm pixels
        """
        for dataset in self._warm_pixels.datasets:
            filename = ut.output_path / f"plotted_distributions/{dataset.name}_plotted_distributions_{self._warm_pixels.all_quadrants_string}.png"
            if _check_path(filename):
                continue

            try:
                warm_pixel_distributions(
                    [
                        dataset.quadrant(quadrant)
                        for quadrant
                        in self._warm_pixels.quadrants_string
                    ],
                    save_path=filename,
                )
            except IndexError as e:
                logger.exception(e)

    def stacked_trails(self):
        """
        Plot a tiled set of stacked trails for each dataset
        """
        for dataset in self._warm_pixels.datasets:
            for group in dataset.groups(
                    self._warm_pixels.quadrants_string
            ):
                filename = ut.output_path / f"stacked_trail_plots/{dataset.name}_plotted_stacked_trails_{self._warm_pixels.all_quadrants_string}.png"
                if _check_path(filename):
                    continue

                stacked_trails(
                    use_corrected=False,
                    save_path=filename,
                    group=group,
                )

    def density(self, extension="png"):
        """
        Plot the evolution of trap density over time
        """
        save_path = ut.output_path / f"density_evol_{self.list_name}{self._warm_pixels.quadrants_string}.{extension}"
        if _check_path(save_path):
            return

        trap_density_evol(
            all_trap_densities=self._warm_pixels.all_trap_densities(),
            use_corrected=self.use_corrected,
            save_path=save_path
        )
