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
            # Plot
            plot_warm_pixels(
                image_quadrant.array(),
                PixelLineCollection(
                    image_quadrant.warm_pixels(),
                ),
                save_path=dataset.output_path / image_quadrant.name,
            )


class Plot:
    def __init__(
            self,
            warm_pixels_,
            list_name,
            use_corrected,
            quadrants_string,
    ):
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
        for dataset in self._warm_pixels.datasets:
            plot_all_warm_pixels(dataset)

    def warm_pixel_distributions(self):
        for dataset in self._warm_pixels.datasets:
            filename = ut.output_path / f"plotted_distributions/{dataset.name}_plotted_distributions_{self.all_quadrants_string}.png"
            plot_warm_pixel_distributions(
                dataset.all_quadrants,
                save_path=filename,
            )

    def stacked_trails(self):
        for dataset in self._warm_pixels.datasets:
            for group in dataset.groups:
                filename = ut.output_path / f"stacked_trail_plots/{dataset.name}_plotted_stacked_trails_{self.all_quadrants_string}.png"
                plot_stacked_trails(
                    use_corrected=False,
                    save_path=filename,
                    group=group,
                )

    def density(self, extension="png"):
        save_path = ut.output_path / f"density_evol_{self.list_name}{self.quadrants_string}.{extension}"
        plot_trap_density_evol(
            all_trap_densities=self._warm_pixels.all_trap_densities(),
            use_corrected=self.use_corrected,
            save_path=save_path
        )

    def by_name(self, plot_names):
        for name in plot_names:
            name = name.replace("-", "_")
            if name not in self.all_methods:
                raise OptionException(
                    f"{name} not a valid plot option. Choose from {self.all_methods}"
                )

            print(f"Attempting to plot {name}")
            getattr(self, name)()
            print(f"Plotted {name}")
