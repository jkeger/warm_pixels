import os
from typing import List

from warm_pixels import hst_utilities as ut
from warm_pixels.pixel_lines import PixelLineCollection
from .stacked_trails import plot_stacked_trails
from .trap_density import plot_trap_density_evol
from .warm_pixels import plot_warm_pixels
from .warm_pixels import plot_warm_pixels, plot_warm_pixel_distributions


class OptionException(Exception):
    pass


def plot_all_warm_pixels(dataset):
    for quadrant in dataset.all_quadrants:
        for image_quadrant in quadrant.image_quadrants:
            filename = dataset.output_path / f"{image_quadrant.name}.png"
            if _check_path(filename):
                continue

            # Plot
            plot_warm_pixels(
                image_quadrant.array(),
                PixelLineCollection(
                    image_quadrant.warm_pixels(),
                ),
                save_path=filename,
            )


def _check_path(
        path
):
    if path.exists():
        return True
    os.makedirs(
        path.parent,
        exist_ok=True
    )
    return False


class Plot:
    def __init__(
            self,
            warm_pixels_,
            list_name,
            use_corrected,
            quadrants_string,
    ):
        """
        Handles plotting of various outputs from the pipeline.

        Parameters
        ----------
        warm_pixels_
            API to access pipeline output such as warm pixels and fits
        list_name
            A name for the set of data
        use_corrected
            Are images CTI corrected?
        quadrants_string
            A string describing the groups of quadrants from HST images included.
            e.g. AB_CD where AB and CD are groups
        """
        self._warm_pixels = warm_pixels_
        self.list_name = list_name
        self.use_corrected = use_corrected

        self.quadrants_string = quadrants_string
        self.all_quadrants_string = "".join(
            quadrants_string.split("_")
        )

        self.all_methods = {
            name for name in dir(self)
            if not name.startswith("__")
        }

    def warm_pixels(self):
        """
        Plot warm pixels for each quadrant of each dataset
        """
        for dataset in self._warm_pixels.datasets:
            plot_all_warm_pixels(dataset)

    def warm_pixel_distributions(self):
        """
        Plot a histogram of the distribution of warm pixels
        """
        for dataset in self._warm_pixels.datasets:
            filename = ut.output_path / f"plotted_distributions/{dataset.name}_plotted_distributions_{self.all_quadrants_string}.png"
            if _check_path(filename):
                continue

            plot_warm_pixel_distributions(
                dataset.all_quadrants,
                save_path=filename,
            )

    def stacked_trails(self):
        """
        Plot a tiled set of stacked trails for each dataset
        """
        for dataset in self._warm_pixels.datasets:
            for group in dataset.groups:
                filename = ut.output_path / f"stacked_trail_plots/{dataset.name}_plotted_stacked_trails_{self.all_quadrants_string}.png"
                if _check_path(filename):
                    continue

                plot_stacked_trails(
                    use_corrected=False,
                    save_path=filename,
                    group=group,
                )

    def density(self, extension="png"):
        """
        Plot the evolution of trap density over time
        """
        save_path = ut.output_path / f"density_evol_{self.list_name}{self.quadrants_string}.{extension}"
        if _check_path(save_path):
            return

        plot_trap_density_evol(
            all_trap_densities=self._warm_pixels.all_trap_densities(),
            use_corrected=self.use_corrected,
            save_path=save_path
        )

    def by_name(self, plot_names: List[str]):
        """
        Create all plots in the list of plot names.

        This is so plots can be conveniently passed in via the command line.

        Parameters
        ----------
        plot_names
            A list of names of plots. These should match method names from this class
            but may use hyphens instead of underscores.

            e.g. ["density", "warm-pixel-distributions"]
        """
        for name in plot_names:
            name = name.replace("-", "_")
            if name not in self.all_methods:
                raise OptionException(
                    f"{name} not a valid plot option. Choose from {self.all_methods}"
                )

            print(f"Attempting to plot {name}")
            getattr(self, name)()
            print(f"Plotted {name}")
