from warm_pixels import hst_utilities as ut
from warm_pixels.pixel_lines import PixelLineCollection
from .stacked_trails import plot_stacked_trails
from .trap_density import plot_trap_density_evol
from .warm_pixels import plot_warm_pixels
from .warm_pixels import plot_warm_pixels, plot_warm_pixel_distributions


def plot_all_warm_pixels(quadrant_dataset_):
    for quadrant in quadrant_dataset_.all_quadrants:
        for image_quadrant in quadrant.image_quadrants:
            # Plot
            plot_warm_pixels(
                image_quadrant.array(),
                PixelLineCollection(
                    image_quadrant.warm_pixels(),
                ),
                save_path=quadrant_dataset_.dataset.output_path / image_quadrant.name,
            )


def output_plots(
        warm_pixels_,
        list_name,
        use_corrected=False,
        plot_density=False,
        plot_warm_pixels=False,
):
    for quadrant_dataset_ in warm_pixels_.quadrant_datasets():
        dataset = quadrant_dataset_.dataset

        if plot_warm_pixels:
            plot_all_warm_pixels(quadrant_dataset_)

        filename = dataset.plotted_distributions(
            warm_pixels_.quadrants
        )
        plot_warm_pixel_distributions(
            quadrant_dataset_.all_quadrants,
            save_path=filename,
        )

        for group in quadrant_dataset_.groups:
            filename = dataset.plotted_stacked_trails(
                group,
            )
            plot_stacked_trails(
                use_corrected=False,
                save_path=filename,
                group=group,
            )

    if plot_density:
        save_path = ut.dataset_list_plotted_density_evol(
            list_name,
            [
                trap_densities.quadrants_string
                for trap_densities
                in warm_pixels_.all_trap_densities()
            ]
        )
        print("Plot trap density evolution...", end=" ", flush=True)
        plot_trap_density_evol(
            all_trap_densities=warm_pixels_.all_trap_densities(),
            use_corrected=use_corrected,
            save_path=save_path
        )
